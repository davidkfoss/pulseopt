"""Tests for additive runtime metric derivation in result JSONs."""

from __future__ import annotations

import math

import pytest

from experiments.utils import results as result_utils


def build_result_payload(**overrides: object) -> dict[str, object]:
    """Create a compact run-result payload for runtime metric tests."""

    payload: dict[str, object] = {
        "method_name": "AdaptiveScheduler",
        "task_name": "sst2",
        "seed": 0,
        "model_name": "distilbert-base-uncased",
        "dataset_name": "GLUE SST-2",
        "config": {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "device": "cpu",
            "num_workers": 0,
            "pin_memory": False,
            "run_tag": "runtime_benchmark",
            "structured_control_mode": "independent",
            "context_mode": "none",
            "lr_scheduler": "none",
            "episode_length": 200,
            "lr_candidates": [1.0],
            "noise_candidates": [0.0, 0.005, 0.01],
            "control_regime": "AdaptiveScheduler",
        },
        "best_val_accuracy": 0.8,
        "final_val_accuracy": 0.78,
        "final_train_loss": 0.5,
        "total_steps": 100,
        "total_epochs": 5,
        "wall_clock_seconds": 20.0,
        "estimated_flops": 4e12,
        "accuracy_per_tflop": 0.2,
        "train_losses": [0.6, 0.5],
        "train_accuracies": [0.7, 0.8],
        "val_accuracies": [0.75, 0.78],
        "epoch_wall_clock_seconds": [3.0, 4.0, 5.0, 4.0, 4.0],
        "epoch_tflops": [0.5, 0.75, 0.9, 0.85, 1.0],
        "episode_logs": {},
        "controller_logs": {
            "_logged_episode_count": 5,
            "lr_controller_logs": {
                "axis_name": "lr",
                "controller_type": "FixedAxis",
                "controller_created": False,
                "fixed_arm_value": 1.0,
                "arm_values": [1.0],
            },
            "noise_controller_logs": {
                "axis_name": "noise",
                "controller_type": "DiscountedUCBController",
                "controller_created": True,
                "arm_values": [0.0, 0.005, 0.01],
                "controller_updates_history": [1, 2, 3, 4, 5],
                "value_estimates_history": [
                    [0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                ],
            },
        },
    }
    payload.update(overrides)
    return payload


def test_runtime_metrics_preserve_existing_top_level_wall_clock_seconds() -> None:
    """The additive runtime block should not alter existing top-level timing fields."""

    payload = build_result_payload()
    result = result_utils.result_to_dict(payload)

    assert result["wall_clock_seconds"] == 20.0
    assert result["runtime_metrics"]["wall_clock_seconds"] == 20.0
    assert result["run_tag"] == "runtime_benchmark"


def test_runtime_metrics_compute_step_and_example_throughput() -> None:
    """Derived step/example throughput should match the documented formulas."""

    payload = build_result_payload()
    result = result_utils.result_to_dict(payload)
    metrics = result["runtime_metrics"]

    assert metrics["seconds_per_step"] == 0.2
    assert metrics["steps_per_second"] == 5.0
    assert metrics["effective_batch_size"] == 32
    assert metrics["estimated_train_examples_processed"] == 3200
    assert metrics["estimated_examples_per_second"] == 160.0
    assert metrics["estimated_tflops_total"] == 4.0
    assert metrics["estimated_tflops_per_second"] == 0.2
    assert metrics["estimated_tflops_per_hour"] == 720.0


def test_runtime_metrics_handle_missing_optional_fields_with_nulls() -> None:
    """Missing optional timing and controller fields should map to null-like None values."""

    payload = build_result_payload(
        total_steps=0,
        total_epochs=0,
        epoch_wall_clock_seconds=None,
        epoch_tflops=None,
        estimated_flops=None,
        controller_logs=None,
    )
    payload["config"] = {"batch_size": 8}

    result = result_utils.result_to_dict(payload)
    metrics = result["runtime_metrics"]

    assert metrics["seconds_per_step"] is None
    assert metrics["steps_per_second"] is None
    assert metrics["seconds_per_epoch"] is None
    assert metrics["epochs_per_hour"] is None
    assert metrics["mean_epoch_wall_clock_seconds"] is None
    assert metrics["estimated_tflops_total"] is None
    assert metrics["logged_episode_count"] is None
    assert metrics["controller_update_count_by_axis"] is None


def test_runtime_metrics_collect_hardware_without_cuda(monkeypatch) -> None:
    """Hardware collection should succeed cleanly on CPU-only setups."""

    monkeypatch.setattr(result_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(result_utils.torch.cuda, "device_count", lambda: 0)

    hardware = result_utils.build_runtime_metrics(build_result_payload())["hardware"]

    assert hardware["cuda_available"] is False
    assert hardware["gpu_count"] == 0
    assert hardware["current_device_index"] is None
    assert hardware["current_device_name"] is None


def test_runtime_metrics_infer_fixed_axis_optimization_from_controller_logs() -> None:
    """AEES overhead metadata should report adaptive/fixed axes and controller creation."""

    result = result_utils.result_to_dict(build_result_payload())
    overhead = result["runtime_metrics"]["aees_overhead_metadata"]
    history = result["runtime_metrics"]["controller_history_length_by_axis"]
    updates = result["runtime_metrics"]["controller_update_count_by_axis"]

    assert overhead["fixed_axis_optimization_enabled"] is True
    assert overhead["controller_created_by_axis"] == {"lr": False, "noise": True}
    assert overhead["num_controllers_created"] == 1
    assert overhead["num_adaptive_axes"] == 1
    assert overhead["num_fixed_axes"] == 1
    assert overhead["adaptive_axes"] == ["noise=[0, 0.005, 0.01]"]
    assert overhead["fixed_axes"] == ["lr=1"]
    assert updates == {"lr": 0, "noise": 5}
    assert history == {"lr": 0, "noise": 2}


def test_runtime_metrics_epoch_statistics_use_population_stddev() -> None:
    """Epoch timing aggregates should match the stored epoch list."""

    result = result_utils.result_to_dict(build_result_payload())
    metrics = result["runtime_metrics"]

    assert metrics["mean_epoch_wall_clock_seconds"] == 4.0
    assert metrics["std_epoch_wall_clock_seconds"] == pytest.approx(math.sqrt(0.4))
    assert metrics["min_epoch_wall_clock_seconds"] == 3.0
    assert metrics["max_epoch_wall_clock_seconds"] == 5.0
    assert metrics["first_epoch_wall_clock_seconds"] == 3.0
    assert metrics["last_epoch_wall_clock_seconds"] == 4.0
