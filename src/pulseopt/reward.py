"""Small reward helpers for episode-level scheduling feedback."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from pulseopt.types import EpisodeSummary


class BaseReward(Protocol):
    """Episode-level reward interface."""

    def compute(self, summary: EpisodeSummary) -> float:
        """Compute a scalar reward from one completed episode."""

    def get_last_details(self) -> dict[str, float] | None:
        """Return the most recent reward components when available."""


@dataclass(frozen=True)
class RewardDetails:
    """Structured reward components for logging and debugging."""

    reward_base: float
    reward_instability: float
    reward_penalty: float
    reward_final_unclipped: float
    reward_final_clipped: float


class NormalizedLossImprovementReward:
    """EMA loss improvement reward with an instability penalty and clipping."""

    def __init__(
        self,
        reward_epsilon: float = 1e-8,
        reward_instability_lambda: float = 0.0,
        reward_clip_min: float = -1.0,
        reward_clip_max: float = 1.0,
    ) -> None:
        if reward_epsilon <= 0.0:
            raise ValueError("reward_epsilon must be positive.")
        if reward_instability_lambda < 0.0:
            raise ValueError("reward_instability_lambda must be non-negative.")
        if reward_clip_min > reward_clip_max:
            raise ValueError("reward_clip_min must be <= reward_clip_max.")

        self.reward_epsilon = reward_epsilon
        self.reward_instability_lambda = reward_instability_lambda
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max
        self._last_details: RewardDetails | None = None

    def compute(self, summary: EpisodeSummary) -> float:
        """Return log EMA loss improvement minus an instability penalty."""

        self._validate_summary(summary)
        if summary.ema_loss_start + self.reward_epsilon <= 0.0:
            raise ValueError(
                "ema_loss_start + reward_epsilon must be positive for log reward.")
        if summary.ema_loss_end + self.reward_epsilon <= 0.0:
            raise ValueError(
                "ema_loss_end + reward_epsilon must be positive for log reward.")
        reward_base = math.log(summary.ema_loss_start + self.reward_epsilon) - math.log(
            summary.ema_loss_end + self.reward_epsilon
        )
        mean_step_loss = sum(summary.step_losses) / len(summary.step_losses)
        variance = sum(
            (step_loss - mean_step_loss) ** 2 for step_loss in summary.step_losses
        ) / len(summary.step_losses)
        reward_instability = variance / (
            mean_step_loss * mean_step_loss + self.reward_epsilon
        )
        reward_penalty = self.reward_instability_lambda * reward_instability
        reward_final_unclipped = reward_base - reward_penalty
        reward_final_clipped = min(
            max(reward_final_unclipped, self.reward_clip_min),
            self.reward_clip_max,
        )
        self._last_details = RewardDetails(
            reward_base=reward_base,
            reward_instability=reward_instability,
            reward_penalty=reward_penalty,
            reward_final_unclipped=reward_final_unclipped,
            reward_final_clipped=reward_final_clipped,
        )
        return reward_final_clipped

    def get_last_details(self) -> dict[str, float] | None:
        """Return the most recent reward components for episode logging."""

        if self._last_details is None:
            return None
        return {
            "reward_base": self._last_details.reward_base,
            "reward_instability": self._last_details.reward_instability,
            "reward_penalty": self._last_details.reward_penalty,
            "reward_final_unclipped": self._last_details.reward_final_unclipped,
            "reward_final_clipped": self._last_details.reward_final_clipped,
        }

    def _validate_summary(self, summary: EpisodeSummary) -> None:
        finite_values = {
            "ema_loss_start": summary.ema_loss_start,
            "ema_loss_end": summary.ema_loss_end,
        }
        for name, value in finite_values.items():
            if value is None:
                raise ValueError(f"{name} must be populated.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if summary.steps <= 0:
            raise ValueError(
                "Episode summary must contain at least one completed step.")
        if not summary.step_losses:
            raise ValueError("Episode summary must contain raw step losses.")
        for step_loss in summary.step_losses:
            if not math.isfinite(step_loss):
                raise ValueError(
                    "step_losses must contain only finite values.")
