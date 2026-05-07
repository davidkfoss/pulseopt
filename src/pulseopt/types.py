"""Small shared dataclasses used by the scheduler core."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateConfig:
    """Serializable candidate configuration for one scheduler arm."""

    name: str
    lr_multiplier: float = 1.0
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Candidate config name must be a non-empty string.")
        if self.lr_multiplier <= 0.0:
            raise ValueError("lr_multiplier must be positive.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")


@dataclass(frozen=True)
class StructuredSelection:
    """Selected LR/noise pair for one structured-control episode."""

    lr_index: int
    noise_index: int
    lr_value: float
    noise_value: float
    combined_name: str
    context_bucket: str | None = None


@dataclass(frozen=True)
class StepStats:
    """Minimal per-step bookkeeping for controller decisions."""

    step_index: int
    mode_id: int
    reward: float


@dataclass(frozen=True)
class EpisodeSummary:
    """Lightweight summary for one completed episode."""

    episode_index: int
    mode_id: int
    mode_name: str
    start_step: int
    end_step: int
    steps: int
    loss_start: float
    loss_end: float
    ema_abs_loss_delta: float
    ema_loss_start: float | None = None
    ema_loss_end: float | None = None
    step_losses: tuple[float, ...] = field(default_factory=tuple)
    reward: float | None = None
    mean_update_norm: float | None = None
    reward_base: float | None = None
    reward_instability: float | None = None
    reward_penalty: float | None = None
    reward_final_unclipped: float | None = None
    reward_final_clipped: float | None = None
