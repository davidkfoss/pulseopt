"""Episode managers for AEES candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pulseopt.controller import BaseController
from pulseopt.modes import build_generated_candidate_name
from pulseopt.reward import BaseReward
from pulseopt.tracking import EMATracker
from pulseopt.types import CandidateConfig, EpisodeSummary, StructuredSelection


@dataclass
class _ActiveStructuredEpisode:
    """Mutable state for the currently open structured-control episode."""

    episode_index: int
    selection: StructuredSelection
    candidate_config: CandidateConfig
    start_step: int
    steps_completed: int = 0
    start_loss: float | None = None
    end_loss: float | None = None
    last_step: int | None = None
    update_norm_sum: float = 0.0
    update_norm_count: int = 0
    step_losses: list[float] = field(default_factory=list)


def _compute_episode_loss_ema(step_losses: list[float], alpha: float) -> tuple[float, float]:
    """Return the first and last values of the episode-local EMA loss trace."""

    if not step_losses:
        raise ValueError("step_losses must contain at least one value.")
    ema_loss = float(step_losses[0])
    ema_loss_start = ema_loss
    for step_loss in step_losses[1:]:
        ema_loss = alpha * float(step_loss) + (1.0 - alpha) * ema_loss
    return ema_loss_start, ema_loss


def _extract_reward_details(reward_fn: BaseReward) -> dict[str, float]:
    """Return the latest reward breakdown when the reward object exposes it."""

    details_getter = getattr(reward_fn, "get_last_details", None)
    if not callable(details_getter):
        return {}
    details = details_getter()
    if details is None:
        return {}
    return details


def _set_controller_context(controller: BaseController | object, bucket_id: str | None) -> None:
    """Apply one shared context bucket when the controller supports it."""

    if bucket_id is None:
        return
    context_setter = getattr(controller, "set_context", None)
    if callable(context_setter):
        context_setter(bucket_id)


class StructuredEpisodeManager:
    """Episode manager for AEES structured LR/noise control.

    Holds the LR multiplier and gradient-noise std fixed across one
    fixed-length episode, then routes the episode reward to the active
    LR/noise controllers.  Single-candidate axes are constants and skip
    controller updates.
    """

    def __init__(
        self,
        lr_candidates: list[float],
        noise_candidates: list[float],
        lr_controller: BaseController | None,
        noise_controller: BaseController | list[BaseController] | None,
        reward_fn: BaseReward,
        episode_length: int,
        structured_control_mode: str = "independent",
        context_mode: str = "none",
        total_training_steps: int | None = None,
        context_trend_window: int = 3,
        context_trend_epsilon: float = 1e-3,
        ema_alpha: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        if not lr_candidates:
            raise ValueError("lr_candidates must contain at least one value.")
        if not noise_candidates:
            raise ValueError("noise_candidates must contain at least one value.")
        if episode_length <= 0:
            raise ValueError("episode_length must be a positive integer.")
        if structured_control_mode not in {"independent", "conditional"}:
            raise ValueError("structured_control_mode must be 'independent' or 'conditional'.")
        if context_mode not in {"none", "trend", "trend_phase"}:
            raise ValueError("context_mode must be 'none', 'trend', or 'trend_phase'.")
        if context_trend_window <= 0:
            raise ValueError("context_trend_window must be positive.")
        if context_trend_epsilon < 0.0:
            raise ValueError("context_trend_epsilon must be non-negative.")
        if context_mode == "trend_phase" and (
            total_training_steps is None or total_training_steps <= 0
        ):
            raise ValueError("trend_phase context requires a positive total_training_steps.")
        if len(lr_candidates) == 1:
            if lr_controller is not None:
                raise TypeError("Single-candidate LR axes must not create a controller.")
        elif lr_controller is None:
            raise TypeError("Adaptive LR axes require a controller.")

        if len(noise_candidates) == 1:
            if noise_controller is not None:
                raise TypeError("Single-candidate noise axes must not create a controller.")
        elif structured_control_mode == "independent":
            if noise_controller is None or isinstance(noise_controller, list):
                raise TypeError("independent mode expects one noise controller.")
        else:
            if len(lr_candidates) == 1:
                if noise_controller is None or isinstance(noise_controller, list):
                    raise TypeError(
                        "conditional mode with one LR candidate expects one noise controller."
                    )
            else:
                if not isinstance(noise_controller, list):
                    raise TypeError(
                        "conditional mode expects one noise controller per LR candidate."
                    )
                if len(noise_controller) != len(lr_candidates):
                    raise ValueError(
                        "conditional mode requires exactly one noise controller per LR candidate."
                    )

        self._lr_candidates = [float(value) for value in lr_candidates]
        self._noise_candidates = [float(value) for value in noise_candidates]
        self._lr_controller = lr_controller
        self._noise_controller = noise_controller
        self._reward_fn = reward_fn
        self._episode_length = episode_length
        self._structured_control_mode = structured_control_mode
        self._context_mode = context_mode
        self._total_training_steps = total_training_steps
        self._context_trend_window = context_trend_window
        self._context_trend_epsilon = context_trend_epsilon
        self._ema_alpha = ema_alpha
        self._loss_delta_ema = EMATracker(alpha=ema_alpha, eps=eps)

        self._active_episode: _ActiveStructuredEpisode | None = None
        self._last_loss: float | None = None
        self._next_episode_index = 0
        self._episode_end_loss_history: list[float] = []
        self._logs: dict[str, list[object]] = {
            "selected_lr_indices": [],
            "selected_noise_indices": [],
            "selected_lr_values": [],
            "selected_noise_values": [],
            "selected_combined_names": [],
            "episode_rewards": [],
            "episode_start_steps": [],
            "episode_end_steps": [],
            "episode_start_losses": [],
            "episode_end_losses": [],
            "episode_ema_loss_starts": [],
            "episode_ema_loss_ends": [],
            "reward_base": [],
            "reward_instability": [],
            "reward_penalty": [],
            "reward_final_unclipped": [],
            "reward_final_clipped": [],
            "mean_update_norms": [],
        }
        if context_mode != "none":
            self._logs["context_bucket_ids"] = []
            self._logs["context_bucket_names"] = []
            self._logs["context_trends"] = []
            self._logs["context_phases"] = []

    def on_step_start(self, global_step: int) -> CandidateConfig:
        """Return the active structured config, selecting it only at episode start."""

        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if self._active_episode is None:
            selection = self._select_config(global_step)
            self._active_episode = _ActiveStructuredEpisode(
                episode_index=self._next_episode_index,
                selection=selection,
                candidate_config=CandidateConfig(
                    name=selection.combined_name,
                    lr_multiplier=selection.lr_value,
                    noise_std=selection.noise_value,
                ),
                start_step=global_step,
            )
            self._next_episode_index += 1
        return self._active_episode.candidate_config

    def on_step_end(self, loss: float, update_norm: float | None = None) -> None:
        """Record the latest loss and close the episode when it reaches capacity."""

        if self._active_episode is None:
            raise RuntimeError("on_step_end() requires an active episode from on_step_start().")
        self._update_active_episode(
            loss=loss, update_norm=update_norm, episode=self._active_episode
        )
        if self._active_episode.steps_completed >= self._episode_length:
            self._close_active_episode()

    def finalize(self) -> None:
        """Safely close a trailing partial episode if it has completed any steps."""

        if self._active_episode is None:
            return
        if self._active_episode.steps_completed > 0 and self._active_episode.last_step is not None:
            self._close_active_episode()
        else:
            self._active_episode = None

    def get_logs(self) -> dict[str, list[object]]:
        """Return shallow copies of structured episode traces for logging or tests."""

        return {key: list(values) for key, values in self._logs.items()}

    def _close_active_episode(self) -> None:
        episode = self._active_episode
        if (
            episode is None
            or episode.last_step is None
            or episode.start_loss is None
            or episode.end_loss is None
        ):
            raise RuntimeError("Cannot close an episode without completed steps.")
        combined_mode_id = (
            episode.selection.lr_index * len(self._noise_candidates) + episode.selection.noise_index
        )
        summary = self._build_episode_summary(
            episode_index=episode.episode_index,
            mode_id=combined_mode_id,
            mode_name=episode.selection.combined_name,
            start_step=episode.start_step,
            end_step=episode.last_step,
            steps=episode.steps_completed,
            loss_start=episode.start_loss,
            loss_end=episode.end_loss,
            step_losses=episode.step_losses,
            update_norm_sum=episode.update_norm_sum,
            update_norm_count=episode.update_norm_count,
        )
        reward = self._reward_fn.compute(summary)
        finalized_summary = self._finalize_summary(summary, reward)
        if self._lr_controller is not None:
            self._lr_controller.update(episode.selection.lr_index, reward)
        noise_controller = self._select_noise_controller(episode.selection.lr_index)
        if noise_controller is not None:
            noise_controller.update(
                episode.selection.noise_index,
                reward,
            )
        self._append_structured_logs(finalized_summary, episode.selection)
        self._episode_end_loss_history.append(finalized_summary.loss_end)
        self._active_episode = None

    def _append_structured_logs(
        self,
        summary: EpisodeSummary,
        selection: StructuredSelection,
    ) -> None:
        self._logs["selected_lr_indices"].append(selection.lr_index)
        self._logs["selected_noise_indices"].append(selection.noise_index)
        self._logs["selected_lr_values"].append(selection.lr_value)
        self._logs["selected_noise_values"].append(selection.noise_value)
        self._logs["selected_combined_names"].append(selection.combined_name)
        self._logs["episode_rewards"].append(summary.reward)
        self._logs["episode_start_steps"].append(summary.start_step)
        self._logs["episode_end_steps"].append(summary.end_step)
        self._logs["episode_start_losses"].append(summary.loss_start)
        self._logs["episode_end_losses"].append(summary.loss_end)
        self._logs["episode_ema_loss_starts"].append(summary.ema_loss_start)
        self._logs["episode_ema_loss_ends"].append(summary.ema_loss_end)
        self._logs["reward_base"].append(summary.reward_base)
        self._logs["reward_instability"].append(summary.reward_instability)
        self._logs["reward_penalty"].append(summary.reward_penalty)
        self._logs["reward_final_unclipped"].append(summary.reward_final_unclipped)
        self._logs["reward_final_clipped"].append(summary.reward_final_clipped)
        self._logs["mean_update_norms"].append(summary.mean_update_norm)
        if self._context_mode != "none":
            self._logs["context_bucket_ids"].append(selection.context_bucket_id)
            self._logs["context_bucket_names"].append(selection.context_bucket_name)
            self._logs["context_trends"].append(selection.context_trend)
            self._logs["context_phases"].append(selection.context_phase)

    def _select_config(self, global_step: int) -> StructuredSelection:
        bucket_id, bucket_name, trend, phase = self._resolve_context(global_step)
        if self._lr_controller is None:
            lr_index = 0
        else:
            _set_controller_context(self._lr_controller, bucket_id)
            lr_index = self._lr_controller.select_mode()
            self._validate_index(lr_index, len(self._lr_candidates), "lr_controller")
        noise_controller = self._select_noise_controller(lr_index)
        if noise_controller is None:
            noise_index = 0
        else:
            _set_controller_context(noise_controller, bucket_id)
            noise_index = noise_controller.select_mode()
            self._validate_index(noise_index, len(self._noise_candidates), "noise_controller")
        lr_value = self._lr_candidates[lr_index]
        noise_value = self._noise_candidates[noise_index]
        return StructuredSelection(
            lr_index=lr_index,
            noise_index=noise_index,
            lr_value=lr_value,
            noise_value=noise_value,
            combined_name=build_generated_candidate_name(lr_value, noise_value),
            context_bucket_id=bucket_id,
            context_bucket_name=bucket_name,
            context_trend=trend,
            context_phase=phase,
        )

    def _resolve_context(
        self, global_step: int
    ) -> tuple[str | None, str | None, str | None, str | None]:
        if self._context_mode == "none":
            return None, None, None, None
        trend = self._resolve_context_trend()
        if self._context_mode == "trend":
            return trend, trend, trend, None
        phase = self._resolve_context_phase(global_step)
        bucket_name = f"{phase}_{trend}"
        return bucket_name, bucket_name, trend, phase

    def _resolve_context_trend(self) -> str:
        window_losses = self._episode_end_loss_history[-self._context_trend_window :]
        if len(window_losses) < 2:
            return "stable"
        mean_delta = (window_losses[-1] - window_losses[0]) / (len(window_losses) - 1)
        if mean_delta < -self._context_trend_epsilon:
            return "improving"
        if mean_delta > self._context_trend_epsilon:
            return "worsening"
        return "stable"

    def _resolve_context_phase(self, global_step: int) -> str:
        total_training_steps = self._total_training_steps
        if total_training_steps is None or total_training_steps <= 0:
            return "early"
        progress = min(max(global_step / total_training_steps, 0.0), 1.0)
        if progress < (1.0 / 3.0):
            return "early"
        if progress < (2.0 / 3.0):
            return "middle"
        return "late"

    def _select_noise_controller(self, lr_index: int) -> BaseController | None:
        if self._noise_controller is None:
            return None
        if self._structured_control_mode == "independent":
            if isinstance(self._noise_controller, list):
                raise TypeError("independent mode expects one noise controller.")
            return self._noise_controller
        if isinstance(self._noise_controller, list):
            return self._noise_controller[lr_index]
        return self._noise_controller

    def _build_episode_summary(
        self,
        *,
        episode_index: int,
        mode_id: int,
        mode_name: str,
        start_step: int,
        end_step: int,
        steps: int,
        loss_start: float,
        loss_end: float,
        step_losses: list[float],
        update_norm_sum: float,
        update_norm_count: int,
    ) -> EpisodeSummary:
        ema_abs_loss_delta = 0.0
        if len(step_losses) > 1:
            ema_abs_loss_delta = sum(
                abs(curr - prev) for prev, curr in zip(step_losses, step_losses[1:], strict=False)
            )
            ema_abs_loss_delta /= len(step_losses) - 1
        ema_loss_start, ema_loss_end = _compute_episode_loss_ema(step_losses, self._ema_alpha)
        mean_update_norm = None
        if update_norm_count > 0:
            mean_update_norm = update_norm_sum / update_norm_count
        return EpisodeSummary(
            episode_index=episode_index,
            mode_id=mode_id,
            mode_name=mode_name,
            start_step=start_step,
            end_step=end_step,
            steps=steps,
            loss_start=loss_start,
            loss_end=loss_end,
            ema_abs_loss_delta=ema_abs_loss_delta,
            ema_loss_start=ema_loss_start,
            ema_loss_end=ema_loss_end,
            step_losses=tuple(step_losses),
            mean_update_norm=mean_update_norm,
        )

    def _finalize_summary(self, summary: EpisodeSummary, reward: float) -> EpisodeSummary:
        reward_details = _extract_reward_details(self._reward_fn)
        return EpisodeSummary(
            episode_index=summary.episode_index,
            mode_id=summary.mode_id,
            mode_name=summary.mode_name,
            start_step=summary.start_step,
            end_step=summary.end_step,
            steps=summary.steps,
            loss_start=summary.loss_start,
            loss_end=summary.loss_end,
            ema_abs_loss_delta=summary.ema_abs_loss_delta,
            ema_loss_start=summary.ema_loss_start,
            ema_loss_end=summary.ema_loss_end,
            step_losses=summary.step_losses,
            reward=reward,
            mean_update_norm=summary.mean_update_norm,
            reward_base=reward_details.get("reward_base"),
            reward_instability=reward_details.get("reward_instability"),
            reward_penalty=reward_details.get("reward_penalty"),
            reward_final_unclipped=reward_details.get("reward_final_unclipped"),
            reward_final_clipped=reward_details.get("reward_final_clipped"),
        )

    def _update_active_episode(
        self,
        *,
        loss: float,
        update_norm: float | None,
        episode: _ActiveStructuredEpisode,
    ) -> None:
        if not math.isfinite(loss):
            raise ValueError("loss must be finite.")
        if update_norm is not None and not math.isfinite(update_norm):
            raise ValueError("update_norm must be finite when provided.")
        if self._last_loss is not None:
            self._loss_delta_ema.update(abs(loss - self._last_loss))
        self._last_loss = loss
        if episode.start_loss is None:
            episode.start_loss = loss
        episode.end_loss = loss
        episode.step_losses.append(loss)
        episode.steps_completed += 1
        episode.last_step = episode.start_step + episode.steps_completed - 1
        if update_norm is not None:
            episode.update_norm_sum += update_norm
            episode.update_norm_count += 1

    def _validate_index(self, index: int, size: int, controller_name: str) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{controller_name}.select_mode() must return an integer arm id.")
        if not 0 <= index < size:
            raise ValueError(
                f"{controller_name}.select_mode() returned arm id {index}, "
                f"but valid ids are in [0, {size})."
            )
