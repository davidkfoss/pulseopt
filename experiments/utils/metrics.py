"""Small JSON-serializable containers for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    """Summary of one completed training run."""

    method_name: str
    task_name: str
    seed: int
    model_name: str
    dataset_name: str
    config: dict[str, object]
    best_val_accuracy: float
    final_val_accuracy: float
    final_train_loss: float
    total_steps: int
    total_epochs: int
    wall_clock_seconds: float
    estimated_flops: float | None
    accuracy_per_tflop: float | None
    train_losses: list[float]
    train_accuracies: list[float] | None
    val_accuracies: list[float]
    epoch_wall_clock_seconds: list[float]
    epoch_tflops: list[float] | None
    episode_logs: dict[str, object] | None
    controller_logs: dict[str, object] | None
    run_tag: str | None = None
    runtime_metrics: dict[str, object] | None = None
