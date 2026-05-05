"""Tests for episode reward computation."""

import math

import pytest

from pulseopt.reward import NormalizedLossImprovementReward
from pulseopt.types import EpisodeSummary


def make_summary(
    *,
    loss_start: float = 2.0,
    loss_end: float = 1.0,
    ema_abs_loss_delta: float = 0.5,
    ema_loss_start: float,
    ema_loss_end: float,
    step_losses: tuple[float, ...],
) -> EpisodeSummary:
    """Create a compact episode summary for reward tests."""

    return EpisodeSummary(
        episode_index=0,
        mode_id=0,
        mode_name="base",
        start_step=0,
        end_step=1,
        steps=len(step_losses),
        loss_start=loss_start,
        loss_end=loss_end,
        ema_abs_loss_delta=ema_abs_loss_delta,
        ema_loss_start=ema_loss_start,
        ema_loss_end=ema_loss_end,
        step_losses=step_losses,
    )


def test_reward_is_positive_when_loss_decreases() -> None:
    """Improvement in loss should produce positive reward."""

    reward_fn = NormalizedLossImprovementReward(reward_instability_lambda=0.0)

    reward = reward_fn.compute(
        make_summary(ema_loss_start=2.0, ema_loss_end=1.0, step_losses=(2.0, 1.0))
    )

    assert reward == pytest.approx(math.log(2.0 + 1e-8) - math.log(1.0 + 1e-8))


def test_reward_is_zero_when_loss_is_unchanged() -> None:
    """No net improvement should produce zero reward."""

    reward_fn = NormalizedLossImprovementReward()

    reward = reward_fn.compute(
        make_summary(
            loss_start=1.5,
            loss_end=1.5,
            ema_abs_loss_delta=0.25,
            ema_loss_start=1.5,
            ema_loss_end=1.5,
            step_losses=(1.5, 1.5),
        )
    )

    assert reward == 0.0


def test_reward_is_negative_when_loss_increases() -> None:
    """Worse final loss should produce negative reward."""

    reward_fn = NormalizedLossImprovementReward(reward_instability_lambda=0.0)

    reward = reward_fn.compute(
        make_summary(
            loss_start=1.0,
            loss_end=1.5,
            ema_abs_loss_delta=0.25,
            ema_loss_start=1.0,
            ema_loss_end=1.5,
            step_losses=(1.0, 1.5),
        )
    )

    assert reward < 0.0


def test_penalty_is_subtracted_before_clipping() -> None:
    """The instability penalty should reduce the unclipped reward before clipping."""

    reward_fn = NormalizedLossImprovementReward(
        reward_instability_lambda=0.5,
        reward_clip_min=-10.0,
        reward_clip_max=10.0,
    )

    reward = reward_fn.compute(
        make_summary(ema_loss_start=2.0, ema_loss_end=1.8, step_losses=(0.5, 3.1, 1.8))
    )
    details = reward_fn.get_last_details()

    assert details is not None
    assert details["reward_final_unclipped"] == pytest.approx(
        details["reward_base"] - details["reward_penalty"]
    )
    assert reward == pytest.approx(details["reward_final_clipped"])


def test_instability_penalty_reduces_reward() -> None:
    """Spikier step losses should reduce the final reward via the penalty term."""

    reward_fn = NormalizedLossImprovementReward(reward_instability_lambda=1.0)

    stable_reward = reward_fn.compute(
        make_summary(ema_loss_start=2.0, ema_loss_end=1.8, step_losses=(2.0, 1.8, 1.8))
    )
    unstable_reward = reward_fn.compute(
        make_summary(ema_loss_start=2.0, ema_loss_end=1.8, step_losses=(0.5, 3.1, 1.8))
    )

    assert unstable_reward < stable_reward


def test_reward_is_clipped_after_penalty() -> None:
    """Clipping should apply to the final reward after subtracting the penalty."""

    reward_fn = NormalizedLossImprovementReward(
        reward_instability_lambda=0.0,
        reward_clip_min=-0.25,
        reward_clip_max=0.25,
    )

    reward = reward_fn.compute(
        make_summary(ema_loss_start=2.0, ema_loss_end=0.0, step_losses=(2.0, 0.0))
    )

    assert reward == pytest.approx(0.25)


def test_small_late_stage_regression_remains_finite() -> None:
    """Late-stage tiny losses should stay finite without denominator-collapse pathology."""

    reward_fn = NormalizedLossImprovementReward(
        reward_instability_lambda=0.0,
        reward_clip_min=-1.0,
        reward_clip_max=1.0,
    )

    reward = reward_fn.compute(
        make_summary(
            loss_start=0.001,
            loss_end=0.002,
            ema_abs_loss_delta=0.001,
            ema_loss_start=0.001,
            ema_loss_end=0.002,
            step_losses=(0.001, 0.002),
        )
    )

    assert math.isfinite(reward)
    assert reward > -1.0


def test_reward_rejects_non_finite_inputs() -> None:
    """Structured summaries with invalid numeric values should fail clearly."""

    reward_fn = NormalizedLossImprovementReward()

    with pytest.raises(ValueError, match="ema_loss_start must be finite"):
        reward_fn.compute(
            make_summary(
                loss_start=float("nan"),
                loss_end=1.0,
                ema_abs_loss_delta=1.0,
                ema_loss_start=float("nan"),
                ema_loss_end=1.0,
                step_losses=(1.0, 1.0),
            )
        )
