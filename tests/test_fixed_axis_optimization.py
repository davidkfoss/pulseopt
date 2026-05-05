"""Regression tests for structured single-candidate axis optimization."""

from __future__ import annotations

import torch

from pulseopt.controller import DiscountedUCBController
from experiments import task_cifar100, task_sst2


def make_sst2_config(**overrides: object) -> task_sst2.ExperimentConfig:
    """Build a compact SST-2 config for controller initialization tests."""

    config = dict(task_sst2.DEFAULT_CONFIG)
    config.update(
        {
            "method": "AdaptiveScheduler",
            "seed": 7,
            "output": "unused.json",
        }
    )
    config.update(overrides)
    return task_sst2.ExperimentConfig(**config)


def make_cifar_config(**overrides: object) -> task_cifar100.ExperimentConfig:
    """Build a compact CIFAR config for controller initialization tests."""

    config = dict(task_cifar100.DEFAULT_CONFIG)
    config.update(
        {
            "control_mode": "adaptive",
            "seed": 11,
            "output": "unused.json",
        }
    )
    config.update(overrides)
    return task_cifar100.ExperimentConfig(**config)


def test_sst2_lr_only_initialization_skips_noise_controller(capsys) -> None:
    """LR-only AEES should treat noise as fixed and avoid building its controller."""

    config = make_sst2_config(
        lr_candidates=[0.5, 1.0, 2.0],
        noise_candidates=[0.0],
    )

    components = task_sst2.build_method_components(
        config=config,
        model=torch.nn.Linear(4, 2),
        total_training_steps=8,
    )

    captured = capsys.readouterr().out
    assert "adaptive axes: lr_multiplier=[0.5, 1.0, 2.0]" in captured
    assert "fixed axes: noise_std=0.0" in captured
    assert isinstance(components.lr_controller, DiscountedUCBController)
    assert components.noise_controller is None
    assert components.controller_logs is not None
    assert components.controller_logs["noise_controller_logs"]["controller_type"] == "FixedAxis"
    assert "value_estimates_history" not in components.controller_logs["noise_controller_logs"]

    assert components.episode_manager is not None
    mode = components.episode_manager.on_step_start(global_step=0)
    assert mode.lr_multiplier in {0.5, 1.0, 2.0}
    assert mode.noise_std == 0.0


def test_sst2_noise_only_initialization_skips_lr_controller(capsys) -> None:
    """Noise-only AEES should treat the LR multiplier as fixed at 1.0."""

    config = make_sst2_config(
        lr_candidates=[1.0],
        noise_candidates=[0.0, 0.005, 0.01],
    )

    components = task_sst2.build_method_components(
        config=config,
        model=torch.nn.Linear(4, 2),
        total_training_steps=8,
    )

    captured = capsys.readouterr().out
    assert "adaptive axes: noise_std=[0.0, 0.005, 0.01]" in captured
    assert "fixed axes: lr_multiplier=1.0" in captured
    assert components.lr_controller is None
    assert isinstance(components.noise_controller, DiscountedUCBController)
    assert components.controller_logs is not None
    assert components.controller_logs["lr_controller_logs"]["controller_type"] == "FixedAxis"
    assert "value_estimates_history" not in components.controller_logs["lr_controller_logs"]

    assert components.episode_manager is not None
    mode = components.episode_manager.on_step_start(global_step=0)
    assert mode.lr_multiplier == 1.0
    assert mode.noise_std in {0.0, 0.005, 0.01}


def test_sst2_full_structured_initialization_keeps_both_controllers() -> None:
    """Full AEES should still build the existing adaptive controllers."""

    config = make_sst2_config(
        lr_candidates=[0.5, 1.0, 2.0],
        noise_candidates=[0.0, 0.005, 0.01],
    )

    components = task_sst2.build_method_components(
        config=config,
        model=torch.nn.Linear(4, 2),
        total_training_steps=8,
    )

    assert isinstance(components.lr_controller, DiscountedUCBController)
    assert isinstance(components.noise_controller, DiscountedUCBController)


def test_cifar_noise_only_initialization_skips_lr_controller(capsys) -> None:
    """CIFAR noise-only AEES should also avoid building a fixed LR controller."""

    config = make_cifar_config(
        lr_candidates=[1.0],
        noise_candidates=[0.0, 0.005, 0.01],
    )

    components = task_cifar100.build_method_components(
        config=config,
        model=torch.nn.Linear(4, 2),
        total_training_steps=8,
    )

    captured = capsys.readouterr().out
    assert "adaptive axes: noise_std=[0.0, 0.005, 0.01]" in captured
    assert "fixed axes: lr_multiplier=1.0" in captured
    assert components.lr_controller is None
    assert isinstance(components.noise_controller, DiscountedUCBController)


def test_cifar_fixed_equivalent_initialization_skips_both_controllers(capsys) -> None:
    """The fixed-equivalent structured config should keep both axes constant without controllers."""

    config = make_cifar_config(
        lr_candidates=[1.0],
        noise_candidates=[0.0],
    )

    components = task_cifar100.build_method_components(
        config=config,
        model=torch.nn.Linear(4, 2),
        total_training_steps=8,
    )

    captured = capsys.readouterr().out
    assert "adaptive axes: <none>" in captured
    assert "fixed axes: lr_multiplier=1.0, noise_std=0.0" in captured
    assert components.lr_controller is None
    assert components.noise_controller is None
    assert components.controller_logs is not None
    assert components.controller_logs["lr_controller_logs"]["controller_type"] == "FixedAxis"
    assert components.controller_logs["noise_controller_logs"]["controller_type"] == "FixedAxis"

    assert components.episode_manager is not None
    mode = components.episode_manager.on_step_start(global_step=0)
    assert mode.lr_multiplier == 1.0
    assert mode.noise_std == 0.0
