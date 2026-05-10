"""High-level AEES wrapper for any PyTorch optimizer."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from pulseopt.controller import (
    TREND_CONTEXT_BUCKETS,
    BucketedContextualController,
    DiscountedUCBController,
    RandomController,
)
from pulseopt.episode import StructuredEpisodeManager
from pulseopt.reward import NormalizedLossImprovementReward
from pulseopt.types import CandidateConfig

_VALID_CONTROL_MODES = frozenset({"adaptive", "random"})
_VALID_CONTEXT_MODES = frozenset({"none", "trend"})
_VALID_STRUCTURED_CONTROL_MODES = frozenset({"independent", "conditional"})


class AEES:
    """Adaptive Episodic Exploration Scheduling wrapper.

    Wraps any ``torch.optim.Optimizer`` with episode-level bandit control over
    LR multipliers and gradient-noise levels.  The caller is responsible for
    ``zero_grad`` and ``loss.backward``; ``step_end`` handles
    ``optimizer.step``, ``lr_scheduler.step``, and episode bookkeeping.

    Axes with a single candidate are treated as fixed constants and get no
    controller — passing ``lr_candidates=[1.0]`` keeps the LR multiplier
    disabled, and ``noise_candidates=[0.0]`` keeps gradient noise off.

    Example::

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N)

        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0, 2.0],
            noise_candidates=[0.0, 0.005],
            episode_length=100,
            lr_scheduler=scheduler,
            seed=42,
        )

        for step, batch in enumerate(dataloader):
            aees.step_start(step)
            optimizer.zero_grad()
            loss = criterion(model(batch))
            loss.backward()
            aees.step_end(loss)

        aees.finalize()
        logs = aees.get_logs()
        print(f"Episodes: {len(logs['episode_rewards'])}")
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        lr_candidates: list[float] | tuple[float, ...] = (1.0,),
        noise_candidates: list[float] | tuple[float, ...] = (0.0,),
        episode_length: int = 100,
        lr_scheduler: Any | None = None,
        control_mode: str = "adaptive",
        structured_control_mode: str = "independent",
        context_mode: str = "none",
        context_trend_window: int = 3,
        context_trend_epsilon: float = 1e-3,
        reward_instability_lambda: float = 0.0,
        reward_epsilon: float = 1e-8,
        reward_clip_min: float = -1.0,
        reward_clip_max: float = 1.0,
        ema_alpha: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if control_mode not in _VALID_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {sorted(_VALID_CONTROL_MODES)}, got {control_mode!r}."
            )
        if context_mode not in _VALID_CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {sorted(_VALID_CONTEXT_MODES)}, got {context_mode!r}."
            )
        if structured_control_mode not in _VALID_STRUCTURED_CONTROL_MODES:
            raise ValueError(
                f"structured_control_mode must be one of "
                f"{sorted(_VALID_STRUCTURED_CONTROL_MODES)}, "
                f"got {structured_control_mode!r}."
            )

        lr_list = list(lr_candidates)
        noise_list = list(noise_candidates)

        if not lr_list:
            raise ValueError("lr_candidates must contain at least one value.")
        if any(value <= 0.0 for value in lr_list):
            raise ValueError("lr_candidates values must all be positive.")
        if not noise_list:
            raise ValueError("noise_candidates must contain at least one value.")
        if any(value < 0.0 for value in noise_list):
            raise ValueError("noise_candidates values must all be non-negative.")
        if episode_length < 1:
            raise ValueError("episode_length must be a positive integer.")

        reward_fn = NormalizedLossImprovementReward(
            reward_epsilon=reward_epsilon,
            reward_instability_lambda=reward_instability_lambda,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )

        bucket_names = list(TREND_CONTEXT_BUCKETS) if context_mode == "trend" else None

        lr_controller = _build_controller(
            n_arms=len(lr_list),
            control_mode=control_mode,
            bucket_names=bucket_names,
            seed=seed,
        )
        noise_controller = _build_noise_controller(
            noise_candidates=noise_list,
            lr_candidates=lr_list,
            control_mode=control_mode,
            structured_control_mode=structured_control_mode,
            bucket_names=bucket_names,
            seed=seed,
        )

        self._episode_manager = StructuredEpisodeManager(
            lr_candidates=lr_list,
            noise_candidates=noise_list,
            lr_controller=lr_controller,
            noise_controller=noise_controller,
            reward_fn=reward_fn,
            episode_length=episode_length,
            structured_control_mode=structured_control_mode,
            context_mode=context_mode,
            context_trend_window=context_trend_window,
            context_trend_epsilon=context_trend_epsilon,
            ema_alpha=ema_alpha,
        )

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._seed = seed
        self._noise_generators: dict[str, torch.Generator] = {}
        self._step_started = False
        self._current_candidate: CandidateConfig | None = None
        self._param_snapshots: list[Tensor] = []

    # ------------------------------------------------------------------
    # Training-loop interface
    # ------------------------------------------------------------------

    def step_start(self, global_step: int) -> CandidateConfig:
        """Select a candidate config for this step.

        Call once per training step, before ``optimizer.zero_grad()``.
        Returns the active :class:`~pulseopt.types.CandidateConfig`
        for inspection or logging; the return value can be ignored.
        """
        if self._step_started:
            raise RuntimeError("step_start() called twice without an intervening step_end().")
        candidate = self._episode_manager.on_step_start(global_step)
        self._param_snapshots = self._capture_param_snapshots()
        self._current_candidate = candidate
        self._step_started = True
        return candidate

    def step_end(self, loss: float | Tensor) -> None:
        """Apply optimizer and scheduler steps, then update episode reward.

        Call once per training step, after ``loss.backward()`` (and after any
        optional gradient clipping).  Internally calls ``optimizer.step()``
        and, if a scheduler was provided, ``lr_scheduler.step()``.

        Raises:
            ValueError: if ``loss`` is NaN or Inf.  Mixed-precision callers
                that expect occasional non-finite losses during loss-scaling
                backoff should guard the call themselves and skip the step.
            RuntimeError: if called without a preceding :meth:`step_start`.
        """
        if not self._step_started:
            raise RuntimeError("step_end() called without a preceding step_start().")
        candidate = self._current_candidate
        if candidate is None:
            raise AssertionError

        float_loss = float(loss.item()) if isinstance(loss, Tensor) else float(loss)
        if not math.isfinite(float_loss):
            raise ValueError("loss passed to step_end() must be finite.")

        base_lrs = [float(g["lr"]) for g in self._optimizer.param_groups]

        for group, base_lr in zip(self._optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr * candidate.lr_multiplier

        if candidate.noise_std > 0.0:
            self._apply_gradient_noise(candidate.noise_std)

        self._optimizer.step()

        for group, base_lr in zip(self._optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base_lr

        update_norm = self._compute_update_norm(self._param_snapshots)

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        self._episode_manager.on_step_end(float_loss, update_norm=update_norm)

        self._step_started = False
        self._current_candidate = None
        self._param_snapshots = []

    def finalize(self) -> None:
        """Close the trailing partial episode after training ends.

        Call once after the training loop completes.
        """
        if self._step_started:
            raise RuntimeError("finalize() called with an open step — call step_end() first.")
        self._episode_manager.finalize()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_logs(self) -> dict[str, list[object]]:
        """Return shallow copies of the episode and controller logs."""
        return self._episode_manager.get_logs()

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """The underlying optimizer (for zero_grad, state_dict, etc.)."""
        return self._optimizer

    @property
    def current_candidate(self) -> CandidateConfig | None:
        """The candidate config selected for the in-flight step, or ``None``.

        Populated by :meth:`step_start` and cleared by :meth:`step_end`.
        Useful for per-step logging without storing the return value of
        :meth:`step_start` yourself.
        """
        return self._current_candidate

    @property
    def episode_manager(self) -> StructuredEpisodeManager:
        """The underlying episode manager for advanced introspection."""
        return self._episode_manager

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_param_snapshots(self) -> list[Tensor]:
        snapshots: list[Tensor] = []
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                snapshots.append(param.detach().clone())
        return snapshots

    def _compute_update_norm(self, snapshots: list[Tensor]) -> float | None:
        if not snapshots:
            return None
        squared_norm = 0.0
        snapshot_index = 0
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                prev = snapshots[snapshot_index]
                snapshot_index += 1
                delta = param.detach() - prev
                squared_norm += float(delta.pow(2).sum().item())
        return squared_norm**0.5

    def _apply_gradient_noise(self, noise_std: float) -> None:
        for group_index, group in enumerate(self._optimizer.param_groups):
            for param_index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                generator = self._get_noise_generator(group_index, param_index, param.grad.device)
                noise = torch.randn(
                    param.grad.shape,
                    generator=generator,
                    device=param.grad.device,
                    dtype=param.grad.dtype,
                )
                param.grad.add_(noise * noise_std)

    def _get_noise_generator(
        self,
        group_index: int,
        param_index: int,
        device: torch.device,
    ) -> torch.Generator:
        key = f"{device}:{group_index}:{param_index}"
        if key not in self._noise_generators:
            generator_device = "cuda" if device.type == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device)
            if self._seed is not None:
                generator.manual_seed(self._seed + group_index * 1009 + param_index)
            self._noise_generators[key] = generator
        return self._noise_generators[key]


# ------------------------------------------------------------------
# Module-level helpers (not part of public API)
# ------------------------------------------------------------------


def _build_controller(
    n_arms: int,
    control_mode: str,
    bucket_names: list[str] | None,
    seed: int | None,
) -> DiscountedUCBController | BucketedContextualController | RandomController | None:
    if n_arms == 1:
        return None
    if control_mode == "random":
        return RandomController(n_arms=n_arms, random_seed=seed)
    if bucket_names is not None:
        return BucketedContextualController(
            n_arms=n_arms,
            bucket_names=bucket_names,
            random_seed=seed,
            prior_from_global=True,
        )
    return DiscountedUCBController(n_arms=n_arms, random_seed=seed)


def _build_noise_controller(
    noise_candidates: list[float],
    lr_candidates: list[float],
    control_mode: str,
    structured_control_mode: str,
    bucket_names: list[str] | None,
    seed: int | None,
) -> object | list[object] | None:
    if len(noise_candidates) == 1:
        return None

    if structured_control_mode == "independent" or len(lr_candidates) == 1:
        return _build_controller(
            n_arms=len(noise_candidates),
            control_mode=control_mode,
            bucket_names=bucket_names,
            seed=None if seed is None else seed + 101,
        )

    # conditional mode with multiple LR arms: one noise controller per LR arm
    return [
        _build_controller(
            n_arms=len(noise_candidates),
            control_mode=control_mode,
            bucket_names=bucket_names,
            seed=None if seed is None else seed + 101 + lr_index,
        )
        for lr_index in range(len(lr_candidates))
    ]
