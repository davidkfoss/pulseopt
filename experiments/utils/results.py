"""Tiny JSON persistence helpers for experiment results."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform as platform_lib
import socket
import statistics
import sys
from typing import Any

import torch


def result_to_dict(result: object) -> dict[str, Any]:
    """Convert a dataclass-like result object into plain JSON data."""

    converted = _to_jsonable(result)
    if not isinstance(converted, dict):
        raise TypeError("result_to_dict expects a dataclass or dict-like result object.")
    if "run_tag" not in converted:
        converted["run_tag"] = _extract_run_tag(converted)
    converted["runtime_metrics"] = build_runtime_metrics(converted)
    return converted


def save_run_result(result: object, output_path: str | Path) -> None:
    """Persist one run result as stable, indented JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _to_jsonable(value: object) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def build_runtime_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Derive additive runtime and throughput metadata from one run result."""

    wall_clock_seconds = _as_float(result.get("wall_clock_seconds"))
    epoch_wall_clock_seconds = _float_list_or_none(result.get("epoch_wall_clock_seconds"))
    total_steps = _as_int(result.get("total_steps"))
    total_epochs = _as_int(result.get("total_epochs"))
    estimated_flops = _as_float(result.get("estimated_flops"))
    epoch_tflops = _float_list_or_none(result.get("epoch_tflops"))
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    controller_logs = (
        result.get("controller_logs") if isinstance(result.get("controller_logs"), dict) else None
    )

    wall_clock_minutes = wall_clock_seconds / 60.0 if wall_clock_seconds is not None else None
    wall_clock_hours = wall_clock_seconds / 3600.0 if wall_clock_seconds is not None else None

    batch_size = _as_int(config.get("batch_size"))
    gradient_accumulation_steps = _as_int(config.get("gradient_accumulation_steps"))
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1
    effective_batch_size = None
    if batch_size is not None:
        effective_batch_size = batch_size * gradient_accumulation_steps

    estimated_train_examples_processed = None
    if total_steps is not None and effective_batch_size is not None:
        estimated_train_examples_processed = total_steps * effective_batch_size

    estimated_tflops_total = None
    if estimated_flops is not None:
        estimated_tflops_total = estimated_flops / 1e12

    seconds_per_step = None
    steps_per_second = None
    if wall_clock_seconds is not None and total_steps is not None and total_steps > 0:
        seconds_per_step = wall_clock_seconds / total_steps
        steps_per_second = _ratio_or_none(total_steps, wall_clock_seconds)

    seconds_per_epoch = None
    epochs_per_hour = None
    if wall_clock_seconds is not None and total_epochs is not None and total_epochs > 0:
        seconds_per_epoch = wall_clock_seconds / total_epochs
        epochs_per_hour = _ratio_or_none(total_epochs * 3600.0, wall_clock_seconds)

    runtime_metrics: dict[str, Any] = {
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_minutes": wall_clock_minutes,
        "wall_clock_hours": wall_clock_hours,
        "epoch_wall_clock_seconds": epoch_wall_clock_seconds,
        "mean_epoch_wall_clock_seconds": _mean_or_none(epoch_wall_clock_seconds),
        "std_epoch_wall_clock_seconds": _pstdev_or_none(epoch_wall_clock_seconds),
        "min_epoch_wall_clock_seconds": min(epoch_wall_clock_seconds) if epoch_wall_clock_seconds else None,
        "max_epoch_wall_clock_seconds": max(epoch_wall_clock_seconds) if epoch_wall_clock_seconds else None,
        "first_epoch_wall_clock_seconds": epoch_wall_clock_seconds[0] if epoch_wall_clock_seconds else None,
        "last_epoch_wall_clock_seconds": epoch_wall_clock_seconds[-1] if epoch_wall_clock_seconds else None,
        "total_steps": total_steps,
        "seconds_per_step": seconds_per_step,
        "steps_per_second": steps_per_second,
        "total_epochs": total_epochs,
        "seconds_per_epoch": seconds_per_epoch,
        "epochs_per_hour": epochs_per_hour,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "estimated_train_examples_processed": estimated_train_examples_processed,
        "estimated_examples_per_second": _ratio_or_none(
            estimated_train_examples_processed,
            wall_clock_seconds,
        ),
        "estimated_flops": estimated_flops,
        "estimated_tflops_total": estimated_tflops_total,
        "estimated_tflops_per_second": _ratio_or_none(estimated_tflops_total, wall_clock_seconds),
        "estimated_tflops_per_hour": _ratio_or_none(
            estimated_tflops_total * 3600.0 if estimated_tflops_total is not None else None,
            wall_clock_seconds,
        ),
        "epoch_tflops": epoch_tflops,
        "mean_epoch_tflops": _mean_or_none(epoch_tflops),
        "hardware": _collect_hardware_metadata(config),
        "aees_overhead_metadata": _build_aees_overhead_metadata(result, controller_logs),
        "logged_episode_count": _logged_episode_count(controller_logs),
        "controller_update_count_by_axis": _controller_metric_by_axis(
            controller_logs,
            metric_name="update_count",
        ),
        "controller_select_count_by_axis": None,
        "controller_history_length_by_axis": _controller_metric_by_axis(
            controller_logs,
            metric_name="history_length",
        ),
    }
    return runtime_metrics


def _extract_run_tag(result: dict[str, Any]) -> str | None:
    config = result.get("config")
    if isinstance(config, dict):
        run_tag = config.get("run_tag")
        if isinstance(run_tag, str) or run_tag is None:
            return run_tag
    return None


def _collect_hardware_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Collect lightweight runtime environment metadata without failing on CPU-only runs."""

    cuda_available = bool(torch.cuda.is_available())
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names: list[str] = []
    gpu_total_memory_mb: list[int] = []
    if cuda_available:
        for device_index in range(gpu_count):
            try:
                properties = torch.cuda.get_device_properties(device_index)
            except Exception:
                gpu_names.append(f"cuda:{device_index}")
                gpu_total_memory_mb.append(0)
                continue
            gpu_names.append(str(getattr(properties, "name", f"cuda:{device_index}")))
            total_memory = getattr(properties, "total_memory", None)
            if isinstance(total_memory, int):
                gpu_total_memory_mb.append(int(total_memory / (1024 * 1024)))
            else:
                gpu_total_memory_mb.append(0)

    current_device_index = None
    current_device_name = None
    if cuda_available and gpu_count > 0:
        try:
            current_device_index = int(torch.cuda.current_device())
            current_device_name = torch.cuda.get_device_name(current_device_index)
        except Exception:
            current_device_index = None
            current_device_name = None

    cudnn_version = None
    try:
        if torch.backends.cudnn.is_available():
            cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    return {
        "hostname": socket.gethostname(),
        "platform": platform_lib.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "current_device_index": current_device_index,
        "current_device_name": current_device_name,
        "device_from_config": config.get("device"),
        "num_workers_from_config": _as_int(config.get("num_workers")),
        "pin_memory_from_config": _as_bool_or_none(config.get("pin_memory")),
        "mixed_precision_from_config": _as_bool_or_none(config.get("mixed_precision")),
        "bf16_from_config": _as_bool_or_none(config.get("bf16")),
        "fp16_from_config": _as_bool_or_none(config.get("fp16")),
    }


def _build_aees_overhead_metadata(
    result: dict[str, Any],
    controller_logs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Infer lightweight AEES overhead metadata from config and controller logs."""

    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    lr_candidates = _float_list_or_none(config.get("lr_candidates"))
    noise_candidates = _float_list_or_none(config.get("noise_candidates"))

    controller_created_by_axis = _controller_created_by_axis(controller_logs)
    adaptive_axes = _axes_from_logs_or_config(
        controller_logs=controller_logs,
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
        adaptive=True,
    )
    fixed_axes = _axes_from_logs_or_config(
        controller_logs=controller_logs,
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
        adaptive=False,
    )

    fixed_axis_optimization_enabled = None
    if any(value is False for value in controller_created_by_axis.values()):
        fixed_axis_optimization_enabled = True
    elif all(value is not None for value in controller_created_by_axis.values()):
        fixed_axis_optimization_enabled = False
    elif (lr_candidates and len(lr_candidates) == 1) or (noise_candidates and len(noise_candidates) == 1):
        fixed_axis_optimization_enabled = True

    return {
        "method_name": result.get("method_name"),
        "control_regime": config.get("control_regime", config.get("control_mode")),
        "structured_control_mode": config.get("structured_control_mode"),
        "context_mode": config.get("context_mode"),
        "lr_scheduler": config.get("lr_scheduler"),
        "episode_length": config.get("episode_length"),
        "lr_candidates": lr_candidates,
        "noise_candidates": noise_candidates,
        "fixed_axis_optimization_enabled": fixed_axis_optimization_enabled,
        "adaptive_axes": adaptive_axes,
        "fixed_axes": fixed_axes,
        "num_adaptive_axes": len(adaptive_axes) if adaptive_axes is not None else None,
        "num_fixed_axes": len(fixed_axes) if fixed_axes is not None else None,
        "num_controllers_created": _num_controllers_created(
            controller_logs=controller_logs,
            lr_candidates=lr_candidates,
            noise_candidates=noise_candidates,
            structured_control_mode=config.get("structured_control_mode"),
        ),
        "controller_created_by_axis": controller_created_by_axis,
    }


def _axes_from_logs_or_config(
    *,
    controller_logs: dict[str, Any] | None,
    lr_candidates: list[float] | None,
    noise_candidates: list[float] | None,
    adaptive: bool,
) -> list[str] | None:
    axes: list[str] = []
    if controller_logs is not None:
        lr_logs = controller_logs.get("lr_controller_logs")
        noise_logs = controller_logs.get("noise_controller_logs")
        noise_logs_by_lr = controller_logs.get("noise_controller_logs_by_lr")
        lr_axis = _axis_description("lr", lr_logs, adaptive)
        noise_axis = _axis_description("noise", noise_logs_by_lr if isinstance(noise_logs_by_lr, dict) else noise_logs, adaptive)
        if lr_axis is not None:
            axes.append(lr_axis)
        if noise_axis is not None:
            axes.append(noise_axis)
        if axes:
            return axes

    if lr_candidates is not None:
        if (len(lr_candidates) > 1) == adaptive:
            axes.append(_format_axis_summary("lr", lr_candidates))
    if noise_candidates is not None:
        if (len(noise_candidates) > 1) == adaptive:
            axes.append(_format_axis_summary("noise", noise_candidates))
    return axes or None


def _axis_description(
    axis_name: str,
    axis_logs: dict[str, Any] | None,
    adaptive: bool,
) -> str | None:
    if axis_logs is None:
        return None
    created = _controller_created_from_axis_logs(axis_logs)
    if created is None or created != adaptive:
        return None
    if adaptive:
        arm_values = _extract_axis_arm_values(axis_logs)
        return _format_axis_summary(axis_name, arm_values) if arm_values is not None else axis_name
    fixed_value = _extract_fixed_axis_value(axis_logs)
    if fixed_value is None:
        arm_values = _extract_axis_arm_values(axis_logs)
        if arm_values and len(arm_values) == 1:
            fixed_value = arm_values[0]
    if fixed_value is None:
        return axis_name
    return f"{axis_name}={fixed_value:g}"


def _format_axis_summary(axis_name: str, values: list[float] | None) -> str:
    if not values:
        return axis_name
    rendered_values = ", ".join(f"{value:g}" for value in values)
    return f"{axis_name}=[{rendered_values}]"


def _controller_created_by_axis(controller_logs: dict[str, Any] | None) -> dict[str, bool | None]:
    lr_value = None
    noise_value = None
    if controller_logs is not None:
        lr_value = _controller_created_from_axis_logs(controller_logs.get("lr_controller_logs"))
        noise_logs = controller_logs.get("noise_controller_logs_by_lr")
        if isinstance(noise_logs, dict):
            per_controller = [
                _controller_created_from_axis_logs(log)
                for log in noise_logs.values()
                if isinstance(log, dict)
            ]
            present_values = [value for value in per_controller if value is not None]
            if present_values:
                noise_value = any(present_values)
        else:
            noise_value = _controller_created_from_axis_logs(controller_logs.get("noise_controller_logs"))
    return {"lr": lr_value, "noise": noise_value}


def _num_controllers_created(
    *,
    controller_logs: dict[str, Any] | None,
    lr_candidates: list[float] | None,
    noise_candidates: list[float] | None,
    structured_control_mode: object,
) -> int | None:
    if controller_logs is not None:
        controller_count = 0
        lr_logs = controller_logs.get("lr_controller_logs")
        if _controller_created_from_axis_logs(lr_logs):
            controller_count += 1
        noise_logs_by_lr = controller_logs.get("noise_controller_logs_by_lr")
        if isinstance(noise_logs_by_lr, dict):
            controller_count += sum(
                1
                for log in noise_logs_by_lr.values()
                if _controller_created_from_axis_logs(log)
            )
            return controller_count
        if _controller_created_from_axis_logs(controller_logs.get("noise_controller_logs")):
            controller_count += 1
        return controller_count

    if lr_candidates is None or noise_candidates is None:
        return None
    controller_count = 1 if len(lr_candidates) > 1 else 0
    if len(noise_candidates) > 1:
        if structured_control_mode == "conditional" and len(lr_candidates) > 1:
            controller_count += len(lr_candidates)
        else:
            controller_count += 1
    return controller_count


def _controller_metric_by_axis(
    controller_logs: dict[str, Any] | None,
    *,
    metric_name: str,
) -> dict[str, int | None] | None:
    if controller_logs is None:
        return None
    return {
        "lr": _axis_metric(controller_logs.get("lr_controller_logs"), metric_name=metric_name),
        "noise": _noise_axis_metric(controller_logs, metric_name=metric_name),
    }


def _axis_metric(axis_logs: object, *, metric_name: str) -> int | None:
    if not isinstance(axis_logs, dict):
        return None
    created = _controller_created_from_axis_logs(axis_logs)
    if created is False:
        return 0
    if metric_name == "update_count":
        return _extract_controller_update_count(axis_logs)
    if metric_name == "history_length":
        return _extract_controller_history_length(axis_logs)
    return None


def _noise_axis_metric(
    controller_logs: dict[str, Any],
    *,
    metric_name: str,
) -> int | None:
    noise_logs_by_lr = controller_logs.get("noise_controller_logs_by_lr")
    if isinstance(noise_logs_by_lr, dict):
        per_controller: list[int] = []
        for axis_log in noise_logs_by_lr.values():
            metric = _axis_metric(axis_log, metric_name=metric_name)
            if metric is not None:
                per_controller.append(metric)
        if not per_controller:
            return None
        if metric_name == "history_length":
            return max(per_controller)
        return sum(per_controller)
    return _axis_metric(controller_logs.get("noise_controller_logs"), metric_name=metric_name)


def _controller_created_from_axis_logs(axis_logs: object) -> bool | None:
    if not isinstance(axis_logs, dict):
        return None
    created = axis_logs.get("controller_created")
    if isinstance(created, bool):
        return created
    controller_type = axis_logs.get("controller_type")
    if controller_type == "FixedAxis":
        return False
    if isinstance(controller_type, str):
        return True
    return None


def _extract_axis_arm_values(axis_logs: object) -> list[float] | None:
    if not isinstance(axis_logs, dict):
        return None
    arm_values = _float_list_or_none(axis_logs.get("arm_values"))
    if arm_values:
        return arm_values
    if "noise_controller_logs_by_lr" in axis_logs:
        return None
    return None


def _extract_fixed_axis_value(axis_logs: object) -> float | None:
    if not isinstance(axis_logs, dict):
        return None
    fixed_value = axis_logs.get("fixed_arm_value")
    if isinstance(fixed_value, (int, float)) and not isinstance(fixed_value, bool):
        return float(fixed_value)
    return None


def _extract_controller_update_count(axis_logs: dict[str, Any]) -> int | None:
    update_history = axis_logs.get("controller_updates_history")
    if isinstance(update_history, list) and update_history:
        return _as_int(update_history[-1])
    global_prior_logs = axis_logs.get("global_prior_logs")
    if isinstance(global_prior_logs, dict):
        global_history = global_prior_logs.get("controller_updates_history")
        if isinstance(global_history, list) and global_history:
            return _as_int(global_history[-1])
    return None


def _extract_controller_history_length(axis_logs: dict[str, Any]) -> int | None:
    history = axis_logs.get("value_estimates_history")
    if isinstance(history, list):
        return len(history)
    global_prior_logs = axis_logs.get("global_prior_logs")
    if isinstance(global_prior_logs, dict):
        global_history = global_prior_logs.get("value_estimates_history")
        if isinstance(global_history, list):
            return len(global_history)
        global_updates = global_prior_logs.get("controller_updates_history")
        if isinstance(global_updates, list):
            return len(global_updates)
    updates = axis_logs.get("controller_updates_history")
    if isinstance(updates, list):
        return len(updates)
    return None


def _logged_episode_count(controller_logs: dict[str, Any] | None) -> int | None:
    if controller_logs is None:
        return None
    return _as_int(controller_logs.get("_logged_episode_count"))


def _as_bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _float_list_or_none(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    converted: list[float] = []
    for item in value:
        item_value = _as_float(item)
        if item_value is None:
            return None
        converted.append(item_value)
    return converted


def _ratio_or_none(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _mean_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _pstdev_or_none(values: list[float] | None) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)
