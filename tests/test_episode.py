"""Tests for episode lifecycle management."""

from __future__ import annotations

import math

import pytest

from pulseopt.episode import StructuredEpisodeManager
from pulseopt.reward import NormalizedLossImprovementReward


class StubController:
    """Small deterministic controller for episode tests."""

    def __init__(self, selections: list[int]) -> None:
        self._selections = list(selections)
        self._cursor = 0
        self.updates: list[tuple[int, float]] = []

    def select_mode(self) -> int:
        if self._cursor >= len(self._selections):
            raise RuntimeError("No more stub selections available.")
        mode_id = self._selections[self._cursor]
        self._cursor += 1
        return mode_id

    def update(self, mode_id: int, reward: float) -> None:
        self.updates.append((mode_id, reward))

    def get_state(self) -> dict[str, object]:
        return {"updates": list(self.updates)}


class ContextAwareStubController(StubController):
    """Deterministic controller that records shared context assignments."""

    def __init__(self, selections: list[int]) -> None:
        super().__init__(selections)
        self.contexts: list[str] = []

    def set_context(self, bucket_id: str) -> None:
        self.contexts.append(bucket_id)


def build_reward() -> NormalizedLossImprovementReward:
    """Create a reward with no instability penalty for simpler assertions."""

    return NormalizedLossImprovementReward(reward_instability_lambda=0.0)


def test_independent_structured_selection_is_fixed_within_episode() -> None:
    """Structured independent mode should hold one LR/noise pair through the episode."""

    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.05],
        lr_controller=StubController([1, 0]),
        noise_controller=StubController([0, 1]),
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
    )

    config_a = manager.on_step_start(global_step=0)
    manager.on_step_end(loss=5.0)
    config_b = manager.on_step_start(global_step=1)
    manager.on_step_end(loss=4.0)
    config_c = manager.on_step_start(global_step=2)

    assert (config_a.lr_multiplier, config_a.noise_std) == (1.2, 0.0)
    assert (config_b.lr_multiplier, config_b.noise_std) == (1.2, 0.0)
    assert (config_c.lr_multiplier, config_c.noise_std) == (1.0, 0.05)


def test_conditional_noise_selection_depends_on_selected_lr() -> None:
    """Conditional mode should choose noise from the LR-conditioned controller bank."""

    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.05],
        lr_controller=StubController([1, 0]),
        noise_controller=[StubController([0]), StubController([1])],
        reward_fn=build_reward(),
        episode_length=1,
        structured_control_mode="conditional",
    )

    first = manager.on_step_start(global_step=0)
    manager.on_step_end(loss=5.0)
    second = manager.on_step_start(global_step=1)

    assert (first.lr_multiplier, first.noise_std) == (1.2, 0.05)
    assert (second.lr_multiplier, second.noise_std) == (1.0, 0.0)


def test_structured_controllers_update_only_when_episode_closes() -> None:
    """LR and noise controllers should receive rewards only at episode end."""

    lr_controller = StubController([0])
    noise_controller = StubController([1])
    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.05],
        lr_controller=lr_controller,
        noise_controller=noise_controller,
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
    )

    manager.on_step_start(global_step=0)
    manager.on_step_end(loss=5.0)
    assert lr_controller.updates == []
    assert noise_controller.updates == []

    manager.on_step_start(global_step=1)
    manager.on_step_end(loss=4.0)
    assert len(lr_controller.updates) == 1
    assert len(noise_controller.updates) == 1


def test_single_candidate_lr_axis_uses_fixed_value_without_controller() -> None:
    """A one-value LR axis should stay fixed while the adaptive noise axis still updates."""

    noise_controller = StubController([1])
    manager = StructuredEpisodeManager(
        lr_candidates=[1.0],
        noise_candidates=[0.0, 0.05],
        lr_controller=None,
        noise_controller=noise_controller,
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
    )

    config = manager.on_step_start(global_step=0)
    manager.on_step_end(loss=5.0)
    manager.on_step_start(global_step=1)
    manager.on_step_end(loss=4.0)

    assert config.lr_multiplier == 1.0
    assert config.noise_std == 0.05
    assert noise_controller.updates


def test_single_candidate_noise_axis_uses_fixed_value_without_controller() -> None:
    """A one-value noise axis should stay fixed while the adaptive LR axis still updates."""

    lr_controller = StubController([1])
    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0],
        lr_controller=lr_controller,
        noise_controller=None,
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
    )

    config = manager.on_step_start(global_step=0)
    manager.on_step_end(loss=5.0)
    manager.on_step_start(global_step=1)
    manager.on_step_end(loss=4.0)

    assert config.lr_multiplier == 1.2
    assert config.noise_std == 0.0
    assert lr_controller.updates


def test_structured_logs_are_populated_for_completed_episode() -> None:
    """Structured runs should log the main per-episode diagnostics."""

    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.05],
        lr_controller=StubController([1]),
        noise_controller=StubController([1]),
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
        ema_alpha=1.0,
    )

    manager.on_step_start(global_step=3)
    manager.on_step_end(loss=9.0, update_norm=0.2)
    manager.on_step_start(global_step=4)
    manager.on_step_end(loss=7.0, update_norm=0.4)

    logs = manager.get_logs()
    assert logs["selected_lr_indices"] == [1]
    assert logs["selected_noise_indices"] == [1]
    assert logs["selected_lr_values"] == [1.2]
    assert logs["selected_noise_values"] == [0.05]
    assert logs["selected_combined_names"] == ["lr1p2_n0p05"]
    assert logs["episode_start_steps"] == [3]
    assert logs["episode_end_steps"] == [4]
    assert logs["episode_ema_loss_starts"] == [9.0]
    assert logs["episode_ema_loss_ends"] == [7.0]
    assert logs["reward_final_clipped"][0] == logs["episode_rewards"][0]
    assert logs["mean_update_norms"][0] == pytest.approx(0.3)


def test_trend_context_is_shared_across_independent_controllers() -> None:
    """Trend context should be passed to both LR and noise controllers."""

    lr_controller = ContextAwareStubController([0, 0, 0])
    noise_controller = ContextAwareStubController([0, 0, 0])
    manager = StructuredEpisodeManager(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.05],
        lr_controller=lr_controller,
        noise_controller=noise_controller,
        reward_fn=build_reward(),
        episode_length=1,
        structured_control_mode="independent",
        context_mode="trend",
        context_trend_window=2,
        context_trend_epsilon=0.1,
    )

    for step, loss in enumerate([5.0, 4.0, 4.5]):
        manager.on_step_start(global_step=step)
        manager.on_step_end(loss=loss)

    logs = manager.get_logs()
    assert logs["context_bucket_ids"] == ["stable", "stable", "improving"]
    assert logs["context_bucket_names"] == ["stable", "stable", "improving"]
    assert logs["context_trends"] == ["stable", "stable", "improving"]

    assert lr_controller.contexts == ["stable", "stable", "improving"]
    assert noise_controller.contexts == ["stable", "stable", "improving"]


def test_finalize_closes_partial_structured_episode_safely() -> None:
    """A partial structured episode with completed steps should be finalized cleanly."""

    manager = StructuredEpisodeManager(
        lr_candidates=[1.0],
        noise_candidates=[0.0],
        lr_controller=None,
        noise_controller=None,
        reward_fn=build_reward(),
        episode_length=3,
        structured_control_mode="independent",
    )

    manager.on_step_start(global_step=0)
    manager.on_step_end(loss=4.0)
    manager.on_step_start(global_step=1)
    manager.on_step_end(loss=3.0)
    manager.finalize()

    logs = manager.get_logs()
    assert logs["episode_start_steps"] == [0]
    assert logs["episode_end_steps"] == [1]


def test_episode_ema_endpoints_are_logged_and_used_for_log_reward() -> None:
    """Episode reward should use the logged EMA start/end values in the log reward."""

    manager = StructuredEpisodeManager(
        lr_candidates=[1.0],
        noise_candidates=[0.0],
        lr_controller=None,
        noise_controller=None,
        reward_fn=build_reward(),
        episode_length=2,
        structured_control_mode="independent",
        ema_alpha=1.0,
    )

    manager.on_step_start(global_step=0)
    manager.on_step_end(loss=10.0)
    manager.on_step_start(global_step=1)
    manager.on_step_end(loss=8.0)

    logs = manager.get_logs()
    assert logs["episode_ema_loss_starts"] == [10.0]
    assert logs["episode_ema_loss_ends"] == [8.0]
    assert logs["episode_rewards"][0] == pytest.approx(
        math.log(10.0 + 1e-8) - math.log(8.0 + 1e-8))


def test_invalid_episode_length_raises_clear_error() -> None:
    """Episode length must be strictly positive."""

    with pytest.raises(ValueError, match="episode_length must be a positive integer"):
        StructuredEpisodeManager(
            lr_candidates=[1.0],
            noise_candidates=[0.0],
            lr_controller=StubController([0]),
            noise_controller=StubController([0]),
            reward_fn=build_reward(),
            episode_length=0,
        )
