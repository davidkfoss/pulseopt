"""Simple controller implementations for AEES arm selection."""

from __future__ import annotations

import math
import random
from typing import Protocol

TREND_CONTEXT_BUCKETS = ["improving", "stable", "worsening"]


class BaseController(Protocol):
    """Shared controller interface over candidate-config arms."""

    def select_mode(self) -> int:
        """Choose the next arm identifier."""

    def update(self, mode_id: int, reward: float) -> None:
        """Ingest the reward observed for a previously selected arm."""

    def get_state(self) -> dict[str, object]:
        """Expose minimal internal state for debugging and tests."""


class DiscountedUCBController:
    """Discounted-UCB bandit controller with deterministic tie-breaking."""

    def __init__(
        self,
        n_arms: int,
        exploration_coef: float = 1.0,
        discount: float = 0.95,
        warmup_rounds: int = 1,
        eps: float = 1e-12,
        random_seed: int | None = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")
        if exploration_coef < 0.0:
            raise ValueError("exploration_coef must be non-negative.")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in the interval (0, 1].")
        if warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.n_arms = n_arms
        self.exploration_coef = exploration_coef
        self.discount = discount
        self.warmup_rounds = warmup_rounds
        self.eps = eps
        self.random_seed = random_seed

        self._discounted_counts = [0.0] * n_arms
        self._discounted_reward_sums = [0.0] * n_arms
        self._warmup_counts = [0] * n_arms
        self._total_updates = 0
        self._rng = random.Random(random_seed)

    def select_mode(self) -> int:
        """Select an arm, prioritizing warmup coverage before UCB."""

        pending_warmup = [
            arm_id
            for arm_id, warmup_count in enumerate(self._warmup_counts)
            if warmup_count < self.warmup_rounds
        ]
        if pending_warmup:
            min_count = min(self._warmup_counts[arm_id] for arm_id in pending_warmup)
            candidates = [
                arm_id for arm_id in pending_warmup if self._warmup_counts[arm_id] == min_count
            ]
            return self._break_tie(candidates)

        total_discounted_count = sum(self._discounted_counts)
        log_term = math.log(total_discounted_count + 1.0)
        scores: list[float] = []
        for arm_id in range(self.n_arms):
            count = max(self._discounted_counts[arm_id], self.eps)
            mean_reward = self._discounted_reward_sums[arm_id] / count
            bonus = self.exploration_coef * math.sqrt(log_term / count)
            scores.append(mean_reward + bonus)

        best_score = max(scores)
        candidates = [
            arm_id
            for arm_id, score in enumerate(scores)
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
        ]
        return self._break_tie(candidates)

    def update(self, mode_id: int, reward: float) -> None:
        """Discount old observations, then add the latest arm reward."""

        self._validate_mode_id(mode_id)
        if not math.isfinite(reward):
            raise ValueError("reward must be a finite float.")

        self._discounted_counts = [count * self.discount for count in self._discounted_counts]
        self._discounted_reward_sums = [
            reward_sum * self.discount for reward_sum in self._discounted_reward_sums
        ]
        self._discounted_counts[mode_id] += 1.0
        self._discounted_reward_sums[mode_id] += reward
        self._warmup_counts[mode_id] += 1
        self._total_updates += 1

    def get_state(self) -> dict[str, object]:
        """Return a debug-friendly snapshot of controller state."""

        mean_rewards = []
        for count, reward_sum in zip(
            self._discounted_counts, self._discounted_reward_sums, strict=True
        ):
            divisor = count if count > self.eps else self.eps
            mean_rewards.append(reward_sum / divisor)
        return {
            "n_arms": self.n_arms,
            "exploration_coef": self.exploration_coef,
            "discount": self.discount,
            "warmup_rounds": self.warmup_rounds,
            "total_updates": self._total_updates,
            "counts": list(self._discounted_counts),
            "reward_sums": list(self._discounted_reward_sums),
            "mean_rewards": mean_rewards,
            "warmup_counts": list(self._warmup_counts),
        }

    def copy_state_from(self, state: dict[str, object]) -> None:
        """Restore controller statistics from a compatible state snapshot."""

        counts = state.get("counts")
        reward_sums = state.get("reward_sums")
        warmup_counts = state.get("warmup_counts")
        total_updates = state.get("total_updates")
        if not isinstance(counts, list) or len(counts) != self.n_arms:
            raise ValueError("state['counts'] must match the controller arm count.")
        if not isinstance(reward_sums, list) or len(reward_sums) != self.n_arms:
            raise ValueError("state['reward_sums'] must match the controller arm count.")
        if not isinstance(warmup_counts, list) or len(warmup_counts) != self.n_arms:
            raise ValueError("state['warmup_counts'] must match the controller arm count.")
        if not isinstance(total_updates, int) or total_updates < 0:
            raise ValueError("state['total_updates'] must be a non-negative integer.")

        self._discounted_counts = [float(value) for value in counts]
        self._discounted_reward_sums = [float(value) for value in reward_sums]
        self._warmup_counts = [int(value) for value in warmup_counts]
        self._total_updates = total_updates

    def _break_tie(self, candidates: list[int]) -> int:
        if not candidates:
            raise RuntimeError("Cannot break a tie with no candidates.")
        return self._rng.choice(candidates)

    def _validate_mode_id(self, mode_id: int) -> None:
        if not 0 <= mode_id < self.n_arms:
            raise ValueError(f"mode_id must be in [0, {self.n_arms}), got {mode_id}.")


class RandomController:
    """Uniform random controller over candidate-config arms."""

    def __init__(self, n_arms: int, random_seed: int | None = None) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")
        self.n_arms = n_arms
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    def select_mode(self) -> int:
        """Sample a valid arm uniformly at random."""

        return self._rng.randrange(self.n_arms)

    def update(self, mode_id: int, reward: float) -> None:
        """Accept the common controller API without tracking state."""

        self._validate_mode_id(mode_id)
        if not math.isfinite(reward):
            raise ValueError("reward must be a finite float.")

    def get_state(self) -> dict[str, object]:
        """Return minimal construction metadata for debugging."""

        return {"n_arms": self.n_arms, "random_seed": self.random_seed}

    def _validate_mode_id(self, mode_id: int) -> None:
        if not 0 <= mode_id < self.n_arms:
            raise ValueError(f"mode_id must be in [0, {self.n_arms}), got {mode_id}.")


class BucketedContextualController:
    """Bucketed contextual wrapper around a discounted-UCB controller."""

    def __init__(
        self,
        n_arms: int,
        bucket_names: list[str],
        exploration_coef: float = 1.0,
        discount: float = 0.95,
        warmup_rounds: int = 1,
        eps: float = 1e-12,
        random_seed: int | None = None,
        prior_from_global: bool = True,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")
        if not bucket_names:
            raise ValueError("bucket_names must contain at least one bucket.")
        if len(set(bucket_names)) != len(bucket_names):
            raise ValueError("bucket_names must be unique.")

        self.n_arms = n_arms
        self.bucket_names = list(bucket_names)
        self.random_seed = random_seed
        self.prior_from_global = prior_from_global
        self._active_bucket_id: str | None = None
        self._global_prior = DiscountedUCBController(
            n_arms=n_arms,
            exploration_coef=exploration_coef,
            discount=discount,
            warmup_rounds=warmup_rounds,
            eps=eps,
            random_seed=random_seed,
        )
        self._bucket_controllers: dict[str, DiscountedUCBController] = {}
        self._bucket_visit_counts = {bucket_name: 0 for bucket_name in self.bucket_names}
        self._bucket_reward_sums = {bucket_name: 0.0 for bucket_name in self.bucket_names}
        self._bucket_chosen_arms = {bucket_name: set() for bucket_name in self.bucket_names}

    def set_context(self, bucket_id: str) -> None:
        """Activate one context bucket before selecting the next arm."""

        if bucket_id not in self._bucket_visit_counts:
            raise ValueError(f"Unknown context bucket '{bucket_id}'.")
        if bucket_id not in self._bucket_controllers:
            controller = DiscountedUCBController(
                n_arms=self.n_arms,
                exploration_coef=self._global_prior.exploration_coef,
                discount=self._global_prior.discount,
                warmup_rounds=self._global_prior.warmup_rounds,
                eps=self._global_prior.eps,
                random_seed=self._derive_bucket_seed(bucket_id),
            )
            if self.prior_from_global:
                controller.copy_state_from(self._global_prior.get_state())
            self._bucket_controllers[bucket_id] = controller
        self._active_bucket_id = bucket_id

    def select_mode(self) -> int:
        """Select an arm from the currently active context bucket."""

        controller = self._get_active_bucket_controller()
        mode_id = controller.select_mode()
        self._bucket_chosen_arms[self._active_bucket_id or ""].add(mode_id)
        return mode_id

    def update(self, mode_id: int, reward: float) -> None:
        """Update the active bucket controller and the global prior."""

        controller = self._get_active_bucket_controller()
        self._global_prior.update(mode_id, reward)
        controller.update(mode_id, reward)
        bucket_id = self._active_bucket_id
        if bucket_id is None:
            raise RuntimeError("No active context bucket configured.")
        self._bucket_visit_counts[bucket_id] += 1
        self._bucket_reward_sums[bucket_id] += reward

    def get_state(self) -> dict[str, object]:
        """Expose global and per-bucket controller statistics for logging."""

        bucket_states: dict[str, object] = {}
        for bucket_name in self.bucket_names:
            bucket_controller = self._bucket_controllers.get(bucket_name)
            bucket_states[bucket_name] = (
                bucket_controller.get_state() if bucket_controller is not None else None
            )
        return {
            "n_arms": self.n_arms,
            "bucket_names": list(self.bucket_names),
            "active_bucket_id": self._active_bucket_id,
            "prior_from_global": self.prior_from_global,
            "global_prior_state": self._global_prior.get_state(),
            "bucket_states": bucket_states,
            "bucket_visit_counts": dict(self._bucket_visit_counts),
            "bucket_reward_sums": dict(self._bucket_reward_sums),
            "bucket_distinct_arm_counts": {
                bucket_name: len(chosen_arms)
                for bucket_name, chosen_arms in self._bucket_chosen_arms.items()
            },
        }

    def _get_active_bucket_controller(self) -> DiscountedUCBController:
        if self._active_bucket_id is None:
            raise RuntimeError("set_context() must be called before select_mode() or update().")
        controller = self._bucket_controllers.get(self._active_bucket_id)
        if controller is None:
            raise RuntimeError("Active context bucket is missing an initialized controller.")
        return controller

    def _derive_bucket_seed(self, bucket_id: str) -> int | None:
        if self.random_seed is None:
            return None
        return self.random_seed + 1000 + self.bucket_names.index(bucket_id)
