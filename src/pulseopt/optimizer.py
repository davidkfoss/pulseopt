"""Scheduler-compatible optimizer wrappers with AEES mode scaling."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor

from pulseopt.types import CandidateConfig


class AdaptiveModeAdamW(torch.optim.AdamW):
    """AdamW optimizer that applies AEES multipliers on top of scheduled bases."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
        mode: CandidateConfig | None = None,
        noise_seed: int | None = None,
    ) -> None:
        if not torch.isfinite(torch.tensor(float(lr))):
            raise ValueError("lr must be finite.")
        if lr <= 0.0:
            raise ValueError("lr must be positive.")
        if not torch.isfinite(torch.tensor(float(eps))):
            raise ValueError("eps must be finite.")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        self._mode = self._normalize_mode(mode)
        self._noise_seed = noise_seed
        self._noise_generators: dict[str, torch.Generator] = {}
        self._last_update_norm: float | None = None
        self._scheduled_base_lrs = [float(group["lr"])
                                    for group in self.param_groups]
        self._scheduled_base_weight_decays = [
            float(group["weight_decay"]) for group in self.param_groups
        ]

    def set_mode(self, mode: CandidateConfig) -> None:
        """Set the active AEES candidate configuration."""

        self._mode = self._normalize_mode(mode)

    def step(self, closure: Any = None) -> Any:
        """Run one AdamW step using scheduled bases scaled by the active mode."""

        self._sync_scheduled_base_hparams_from_param_groups()
        param_snapshots = self._capture_param_snapshots()
        self._apply_effective_hparams()
        try:
            self._apply_gradient_noise()
            loss = super().step(closure=closure)
        finally:
            self._restore_scheduled_base_hparams()
        self._last_update_norm = self._compute_update_norm(param_snapshots)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients."""

        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state together with wrapper-level scheduler state."""

        state = super().state_dict()
        state["candidate_config"] = self._serialize_mode()
        state["mode"] = deepcopy(state["candidate_config"])
        state["scheduled_base_lrs"] = list(self._scheduled_base_lrs)
        state["base_lrs"] = list(self._scheduled_base_lrs)
        state["scheduled_base_weight_decays"] = list(
            self._scheduled_base_weight_decays)
        state["base_weight_decays"] = list(self._scheduled_base_weight_decays)
        state["noise_seed"] = self._noise_seed
        state["noise_generator_states"] = {
            key: generator.get_state()
            for key, generator in self._noise_generators.items()
        }
        state["last_update_norm"] = self._last_update_norm
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer and wrapper-level state."""

        state = deepcopy(state_dict)
        optimizer_state = state.pop("optimizer_state", state)
        mode_state = state.pop("candidate_config", state.pop("mode", None))
        scheduled_base_lrs = state.pop(
            "scheduled_base_lrs",
            state.pop("base_lrs", None),
        )
        scheduled_base_weight_decays = state.pop(
            "scheduled_base_weight_decays",
            state.pop("base_weight_decays", None),
        )
        noise_seed = state.pop("noise_seed", None)
        noise_generator_states = state.pop("noise_generator_states", {})
        last_update_norm = state.pop("last_update_norm", None)

        super().load_state_dict(optimizer_state)

        if mode_state is not None:
            self._mode = self._deserialize_mode(mode_state)
        else:
            self._mode = self._normalize_mode(None)

        self._noise_seed = noise_seed
        self._noise_generators = {}
        for key, generator_state in noise_generator_states.items():
            generator = self._make_generator()
            generator.set_state(generator_state)
            self._noise_generators[key] = generator
        self._last_update_norm = last_update_norm

        self._scheduled_base_lrs = self._restore_base_list(
            scheduled_base_lrs,
            current_values=[float(group["lr"]) for group in self.param_groups],
            field_name="scheduled_base_lrs",
        )
        self._scheduled_base_weight_decays = self._restore_base_list(
            scheduled_base_weight_decays,
            current_values=[float(group["weight_decay"])
                            for group in self.param_groups],
            field_name="scheduled_base_weight_decays",
        )
        self._restore_scheduled_base_hparams()

    def compute_update_norm(self) -> float | None:
        """Return the L2 norm of the most recent parameter update."""

        return self._last_update_norm

    def _normalize_mode(
        self,
        mode: CandidateConfig | None,
    ) -> CandidateConfig:
        if mode is None:
            return CandidateConfig(name="base")
        if isinstance(mode, CandidateConfig):
            return mode
        raise TypeError("mode must be a CandidateConfig.")

    def _sync_scheduled_base_hparams_from_param_groups(self) -> None:
        self._scheduled_base_lrs = [float(group["lr"])
                                    for group in self.param_groups]
        self._scheduled_base_weight_decays = [
            float(group["weight_decay"]) for group in self.param_groups
        ]

    def _apply_effective_hparams(self) -> None:
        for group, base_lr, base_weight_decay in zip(
            self.param_groups,
            self._scheduled_base_lrs,
            self._scheduled_base_weight_decays,
            strict=True,
        ):
            group["lr"] = base_lr * self._mode.lr_multiplier
            group["weight_decay"] = base_weight_decay

    def _restore_scheduled_base_hparams(self) -> None:
        for group, base_lr, base_weight_decay in zip(
            self.param_groups,
            self._scheduled_base_lrs,
            self._scheduled_base_weight_decays,
            strict=True,
        ):
            group["lr"] = base_lr
            group["weight_decay"] = base_weight_decay

    def _apply_gradient_noise(self) -> None:
        if self._mode.noise_std <= 0.0:
            return

        for group_index, group in enumerate(self.param_groups):
            for param_index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                generator = self._get_noise_generator(
                    group_index,
                    param_index,
                    parameter.grad.device,
                )

                noise = torch.randn(
                    parameter.grad.shape,
                    generator=generator,
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
                parameter.grad.add_(noise * self._mode.noise_std)

    def _capture_param_snapshots(self) -> list[Tensor]:
        snapshots: list[Tensor] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                snapshots.append(parameter.detach().clone())
        return snapshots

    def _compute_update_norm(self, param_snapshots: list[Tensor]) -> float | None:
        squared_norm = 0.0
        snapshot_index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                previous = param_snapshots[snapshot_index]
                snapshot_index += 1
                delta = parameter.detach() - previous
                squared_norm += float(delta.pow(2).sum().item())
        if snapshot_index == 0:
            return None
        return squared_norm ** 0.5

    def _serialize_mode(self) -> dict[str, float | str]:
        return {
            "name": self._mode.name,
            "lr_multiplier": self._mode.lr_multiplier,
            "noise_std": self._mode.noise_std,
        }

    def _deserialize_mode(self, state: dict[str, Any]) -> CandidateConfig:
        return CandidateConfig(
            name=str(state["name"]),
            lr_multiplier=float(state["lr_multiplier"]),
            noise_std=float(
                state.get("noise_std", state.get("grad_noise_std", 0.0))),
        )

    def _get_noise_generator(
        self,
        group_index: int,
        param_index: int,
        device: torch.device,
    ) -> torch.Generator:
        key = f"{device}:{group_index}:{param_index}"
        if key not in self._noise_generators:
            generator = self._make_generator(device)
            if self._noise_seed is not None:
                generator.manual_seed(
                    self._noise_seed + group_index * 1009 + param_index)
            self._noise_generators[key] = generator
        return self._noise_generators[key]

    def _make_generator(self, device: torch.device) -> torch.Generator:
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        return torch.Generator(device=generator_device)

    def _restore_base_list(
        self,
        stored_values: list[float] | None,
        current_values: list[float],
        field_name: str,
    ) -> list[float]:
        if stored_values is None:
            return list(current_values)
        if len(stored_values) != len(current_values):
            raise ValueError(
                f"{field_name} length does not match optimizer param_groups."
            )
        return [float(value) for value in stored_values]


class AdaptiveModeSGD(torch.optim.SGD):
    """SGD optimizer that applies AEES multipliers on top of scheduled bases."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        mode: CandidateConfig | None = None,
        noise_seed: int | None = None,
    ) -> None:
        if not torch.isfinite(torch.tensor(float(lr))):
            raise ValueError("lr must be finite.")
        if lr <= 0.0:
            raise ValueError("lr must be positive.")
        if not torch.isfinite(torch.tensor(float(momentum))):
            raise ValueError("momentum must be finite.")
        if momentum < 0.0:
            raise ValueError("momentum must be non-negative.")
        if not torch.isfinite(torch.tensor(float(dampening))):
            raise ValueError("dampening must be finite.")
        if dampening < 0.0:
            raise ValueError("dampening must be non-negative.")

        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        self._mode = self._normalize_mode(mode)
        self._noise_seed = noise_seed
        self._noise_generators: dict[str, torch.Generator] = {}
        self._last_update_norm: float | None = None
        self._scheduled_base_lrs = [
            float(group["lr"]) for group in self.param_groups
        ]
        self._scheduled_base_weight_decays = [
            float(group["weight_decay"]) for group in self.param_groups
        ]

    def set_mode(self, mode: CandidateConfig) -> None:
        """Set the active AEES candidate configuration."""

        self._mode = self._normalize_mode(mode)

    def step(self, closure: Any = None) -> Any:
        """Run one SGD step using scheduled bases scaled by the active mode."""

        self._sync_scheduled_base_hparams_from_param_groups()
        param_snapshots = self._capture_param_snapshots()
        self._apply_effective_hparams()
        try:
            self._apply_gradient_noise()
            loss = super().step(closure=closure)
        finally:
            self._restore_scheduled_base_hparams()
        self._last_update_norm = self._compute_update_norm(param_snapshots)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients."""

        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Serialize optimizer state together with wrapper-level scheduler state."""

        state = super().state_dict()
        state["candidate_config"] = self._serialize_mode()
        state["mode"] = deepcopy(state["candidate_config"])
        state["scheduled_base_lrs"] = list(self._scheduled_base_lrs)
        state["base_lrs"] = list(self._scheduled_base_lrs)
        state["scheduled_base_weight_decays"] = list(
            self._scheduled_base_weight_decays
        )
        state["base_weight_decays"] = list(self._scheduled_base_weight_decays)
        state["noise_seed"] = self._noise_seed
        state["noise_generator_states"] = {
            key: generator.get_state()
            for key, generator in self._noise_generators.items()
        }
        state["last_update_norm"] = self._last_update_norm
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer and wrapper-level state."""

        state = deepcopy(state_dict)
        optimizer_state = state.pop("optimizer_state", state)
        mode_state = state.pop("candidate_config", state.pop("mode", None))
        scheduled_base_lrs = state.pop(
            "scheduled_base_lrs",
            state.pop("base_lrs", None),
        )
        scheduled_base_weight_decays = state.pop(
            "scheduled_base_weight_decays",
            state.pop("base_weight_decays", None),
        )
        noise_seed = state.pop("noise_seed", None)
        noise_generator_states = state.pop("noise_generator_states", {})
        last_update_norm = state.pop("last_update_norm", None)

        super().load_state_dict(optimizer_state)

        if mode_state is not None:
            self._mode = self._deserialize_mode(mode_state)
        else:
            self._mode = self._normalize_mode(None)

        self._noise_seed = noise_seed
        self._noise_generators = {}
        for key, generator_state in noise_generator_states.items():
            generator = self._make_generator()
            generator.set_state(generator_state)
            self._noise_generators[key] = generator
        self._last_update_norm = last_update_norm

        self._scheduled_base_lrs = self._restore_base_list(
            scheduled_base_lrs,
            current_values=[float(group["lr"]) for group in self.param_groups],
            field_name="scheduled_base_lrs",
        )
        self._scheduled_base_weight_decays = self._restore_base_list(
            scheduled_base_weight_decays,
            current_values=[
                float(group["weight_decay"]) for group in self.param_groups
            ],
            field_name="scheduled_base_weight_decays",
        )
        self._restore_scheduled_base_hparams()

    def compute_update_norm(self) -> float | None:
        """Return the L2 norm of the most recent parameter update."""

        return self._last_update_norm

    def _normalize_mode(
        self,
        mode: CandidateConfig | None,
    ) -> CandidateConfig:
        if mode is None:
            return CandidateConfig(name="base")
        if isinstance(mode, CandidateConfig):
            return mode
        raise TypeError("mode must be a CandidateConfig.")

    def _sync_scheduled_base_hparams_from_param_groups(self) -> None:
        self._scheduled_base_lrs = [
            float(group["lr"]) for group in self.param_groups
        ]
        self._scheduled_base_weight_decays = [
            float(group["weight_decay"]) for group in self.param_groups
        ]

    def _apply_effective_hparams(self) -> None:
        for group, base_lr, base_weight_decay in zip(
            self.param_groups,
            self._scheduled_base_lrs,
            self._scheduled_base_weight_decays,
            strict=True,
        ):
            group["lr"] = base_lr * self._mode.lr_multiplier
            group["weight_decay"] = base_weight_decay

    def _restore_scheduled_base_hparams(self) -> None:
        for group, base_lr, base_weight_decay in zip(
            self.param_groups,
            self._scheduled_base_lrs,
            self._scheduled_base_weight_decays,
            strict=True,
        ):
            group["lr"] = base_lr
            group["weight_decay"] = base_weight_decay

    def _apply_gradient_noise(self) -> None:
        if self._mode.noise_std <= 0.0:
            return

        for group_index, group in enumerate(self.param_groups):
            for param_index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                generator = self._get_noise_generator(
                    group_index,
                    param_index,
                    parameter.grad.device,
                )

                noise = torch.randn(
                    parameter.grad.shape,
                    generator=generator,
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
                parameter.grad.add_(noise * self._mode.noise_std)

    def _capture_param_snapshots(self) -> list[Tensor]:
        snapshots: list[Tensor] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                snapshots.append(parameter.detach().clone())
        return snapshots

    def _compute_update_norm(self, param_snapshots: list[Tensor]) -> float | None:
        squared_norm = 0.0
        snapshot_index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                previous = param_snapshots[snapshot_index]
                snapshot_index += 1
                delta = parameter.detach() - previous
                squared_norm += float(delta.pow(2).sum().item())
        if snapshot_index == 0:
            return None
        return squared_norm ** 0.5

    def _serialize_mode(self) -> dict[str, float | str]:
        return {
            "name": self._mode.name,
            "lr_multiplier": self._mode.lr_multiplier,
            "noise_std": self._mode.noise_std,
        }

    def _deserialize_mode(self, state: dict[str, Any]) -> CandidateConfig:
        return CandidateConfig(
            name=str(state["name"]),
            lr_multiplier=float(state["lr_multiplier"]),
            noise_std=float(
                state.get("noise_std", state.get("grad_noise_std", 0.0))),
        )

    def _get_noise_generator(
        self,
        group_index: int,
        param_index: int,
        device: torch.device,
    ) -> torch.Generator:
        key = f"{device}:{group_index}:{param_index}"
        if key not in self._noise_generators:
            generator = self._make_generator(device)
            if self._noise_seed is not None:
                generator.manual_seed(
                    self._noise_seed + group_index * 1009 + param_index)
            self._noise_generators[key] = generator
        return self._noise_generators[key]

    def _make_generator(self, device: torch.device) -> torch.Generator:
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        return torch.Generator(device=generator_device)

    def _restore_base_list(
        self,
        stored_values: list[float] | None,
        current_values: list[float],
        field_name: str,
    ) -> list[float]:
        if stored_values is None:
            return list(current_values)
        if len(stored_values) != len(current_values):
            raise ValueError(
                f"{field_name} length does not match optimizer param_groups."
            )
        return [float(value) for value in stored_values]
