"""Run a CIFAR-100 experiment with wrapped AdamW/SGD AEES scheduling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import sys
import time
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pulseopt.controller import (  # noqa: E402
    BucketedContextualController,
    DiscountedUCBController,
    RandomController,
    TREND_CONTEXT_BUCKETS,
    TREND_PHASE_CONTEXT_BUCKETS,
)
from pulseopt.episode import StructuredEpisodeManager  # noqa: E402
from pulseopt.modes import (  # noqa: E402
    build_candidate_configs_from_axes,
    parse_lr_candidates,
    parse_noise_candidates,
)
from pulseopt.optimizer import AdaptiveModeAdamW, AdaptiveModeSGD  # noqa: E402
from pulseopt.reward import NormalizedLossImprovementReward  # noqa: E402
from pulseopt.types import CandidateConfig  # noqa: E402
from experiments.utils.flops import (  # noqa: E402
    FlopAccumulator,
    estimate_resnet18_cifar100_eval_flops_per_batch,
    estimate_resnet18_cifar100_train_flops_per_batch,
)
from experiments.utils.metrics import RunResult  # noqa: E402
from experiments.utils.results import save_run_result  # noqa: E402

if TYPE_CHECKING:
    from torchvision.datasets import CIFAR100


LabelNoiseType = Literal["none", "symmetric", "asymmetric"]
OptimizerName = Literal["AdamW", "SGD"]
ControlMode = Literal["baseline", "adaptive", "random"]
AdaptiveOptimizer = AdaptiveModeAdamW | AdaptiveModeSGD
LABEL_NOISE_TYPES: tuple[LabelNoiseType, ...] = (
    "none",
    "symmetric",
    "asymmetric",
)
CONTROL_MODE_NAMES: dict[ControlMode, str] = {
    "baseline": "Baseline",
    "adaptive": "AdaptiveScheduler",
    "random": "RandomScheduler",
}

CIFAR100_SUPERCLASS_TO_FINE: dict[str, list[int]] = {
    "aquatic_mammals": [4, 30, 55, 72, 95],
    "fish": [1, 32, 67, 73, 91],
    "flowers": [54, 62, 70, 82, 92],
    "food_containers": [9, 10, 16, 28, 61],
    "fruit_and_vegetables": [0, 51, 53, 57, 83],
    "household_electrical_devices": [22, 39, 40, 86, 87],
    "household_furniture": [5, 20, 25, 84, 94],
    "insects": [6, 7, 14, 18, 24],
    "large_carnivores": [3, 42, 43, 88, 97],
    "large_man-made_outdoor_things": [12, 17, 37, 68, 76],
    "large_natural_outdoor_scenes": [23, 33, 49, 60, 71],
    "large_omnivores_and_herbivores": [15, 19, 21, 31, 38],
    "medium_mammals": [34, 63, 64, 66, 75],
    "non-insect_invertebrates": [26, 45, 77, 79, 99],
    "people": [2, 11, 35, 46, 98],
    "reptiles": [27, 29, 44, 78, 93],
    "small_mammals": [36, 50, 65, 74, 80],
    "trees": [47, 52, 56, 59, 96],
    "vehicles_1": [8, 13, 48, 58, 90],
    "vehicles_2": [41, 69, 81, 85, 89],
}


def build_cifar100_asymmetric_label_mapping() -> dict[int, int]:
    """Build fine-label mapping [a,b,c,d,e] -> a->b, ..., e->a per superclass."""

    mapping: dict[int, int] = {}
    for fine_labels in CIFAR100_SUPERCLASS_TO_FINE.values():
        for label_index, fine_label in enumerate(fine_labels):
            mapping[fine_label] = fine_labels[(
                label_index + 1) % len(fine_labels)]
    if set(mapping) != set(range(100)):
        raise ValueError(
            "CIFAR-100 asymmetric label mapping must cover all 100 labels.")
    return mapping


CIFAR100_ASYMMETRIC_LABEL_MAPPING = build_cifar100_asymmetric_label_mapping()


DEFAULT_CONFIG: dict[str, object] = {
    "task_name": "cifar100",
    "model_name": "resnet18",
    "control_mode": "baseline",
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "momentum": 0.9,
    "lr_scheduler": "none",
    "scheduler_t_max": None,
    "warmup_epochs": 0,
    "reward_epsilon": 1e-8,
    "reward_instability_lambda": 0.0,
    "reward_clip_min": -1.0,
    "reward_clip_max": 1.0,
    "episode_length": 200,
    "ema_alpha": 0.1,
    "lr_candidates": [1.0],
    "noise_candidates": [0.0],
    "structured_control_mode": "independent",
    "context_mode": "none",
    "context_trend_window": 3,
    "context_trend_epsilon": 1e-3,
    "label_noise_type": "none",
    "label_noise_rate": 0.0,
    "num_workers": 2,
    "data_dir": "data/cifar100",
    "run_tag": None,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed configuration for one CIFAR-100 run."""

    control_mode: ControlMode
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: OptimizerName
    momentum: float
    lr_scheduler: str
    scheduler_t_max: int | None
    warmup_epochs: int
    reward_epsilon: float
    reward_instability_lambda: float
    reward_clip_min: float
    reward_clip_max: float
    episode_length: int
    lr_candidates: list[float]
    noise_candidates: list[float]
    structured_control_mode: str
    context_mode: str
    context_trend_window: int
    context_trend_epsilon: float
    label_noise_type: LabelNoiseType
    label_noise_rate: float
    seed: int
    output: str
    ema_alpha: float
    num_workers: int
    data_dir: str
    model_name: str = "resnet18"
    task_name: str = "cifar100"
    run_tag: str | None = None


@dataclass
class TrainingComponents:
    """Control-specific objects used during training."""

    optimizer: torch.optim.Optimizer
    episode_manager: StructuredEpisodeManager | None
    method_name: str
    controller_logs: dict[str, object] | None
    structured_control_mode: str | None = None
    lr_candidates: list[float] | None = None
    noise_candidates: list[float] | None = None
    lr_controller: object | None = None
    noise_controller: object | list[object] | None = None


@dataclass(frozen=True)
class LabelNoiseMetadata:
    """Training-label provenance before and after configured corruption."""

    original_targets: list[int]
    noisy_targets: list[int]
    corrupted_mask: list[bool]
    clean_mask: list[bool]
    corrupted_indices: list[int]


class IndexedDataset(Dataset):
    """Wrap a dataset so batches include the original dataset index."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[object, object, int]:
        inputs, target = self.base_dataset[index]
        return inputs, target, index


def parse_args() -> ExperimentConfig:
    """Parse the CLI into a typed experiment config."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-mode",
        choices=["baseline", "adaptive", "random"],
        default=str(DEFAULT_CONFIG["control_mode"]),
        help="Control regime: baseline, adaptive AEES, or random.",
    )
    parser.add_argument("--epochs", type=int,
                        default=int(DEFAULT_CONFIG["epochs"]))
    parser.add_argument("--batch-size", type=int,
                        default=int(DEFAULT_CONFIG["batch_size"]))
    parser.add_argument("--lr", type=float,
                        default=float(DEFAULT_CONFIG["lr"]))
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(DEFAULT_CONFIG["weight_decay"]),
    )
    parser.add_argument(
        "--optimizer",
        choices=["AdamW", "SGD"],
        default=str(DEFAULT_CONFIG["optimizer"]),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=float(DEFAULT_CONFIG["momentum"]),
        help="SGD momentum.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "linear", "warmup_linear"],
        default=str(DEFAULT_CONFIG["lr_scheduler"]),
    )
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=int(DEFAULT_CONFIG["warmup_epochs"]),
    )
    parser.add_argument(
        "--reward-epsilon",
        type=float,
        default=float(DEFAULT_CONFIG["reward_epsilon"]),
    )
    parser.add_argument(
        "--reward-instability-lambda",
        type=float,
        default=float(DEFAULT_CONFIG["reward_instability_lambda"]),
    )
    parser.add_argument(
        "--reward-clip-min",
        type=float,
        default=float(DEFAULT_CONFIG["reward_clip_min"]),
    )
    parser.add_argument(
        "--reward-clip-max",
        type=float,
        default=float(DEFAULT_CONFIG["reward_clip_max"]),
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=int(DEFAULT_CONFIG["episode_length"]),
    )
    parser.add_argument("--lr-candidates", type=str, default=None)
    parser.add_argument("--noise-candidates", type=str, default=None)
    parser.add_argument(
        "--structured-control-mode",
        choices=["independent", "conditional"],
        default=str(DEFAULT_CONFIG["structured_control_mode"]),
    )
    parser.add_argument(
        "--context-mode",
        choices=["none", "trend", "trend_phase"],
        default=str(DEFAULT_CONFIG["context_mode"]),
    )
    parser.add_argument(
        "--context-trend-window",
        type=int,
        default=int(DEFAULT_CONFIG["context_trend_window"]),
    )
    parser.add_argument(
        "--context-trend-epsilon",
        type=float,
        default=float(DEFAULT_CONFIG["context_trend_epsilon"]),
    )
    parser.add_argument(
        "--label-noise-type",
        choices=LABEL_NOISE_TYPES,
        default=str(DEFAULT_CONFIG["label_noise_type"]),
    )
    parser.add_argument(
        "--label-noise-rate",
        type=float,
        default=float(DEFAULT_CONFIG["label_noise_rate"]),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", type=str, default=DEFAULT_CONFIG["run_tag"])
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional output path. Defaults to "
            "results/cifar100_<control_mode>_<optimizer>_seed<seed>.json"
        ),
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.momentum < 0.0:
        raise ValueError("--momentum must be non-negative.")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive.")
    if args.reward_epsilon <= 0.0:
        raise ValueError("--reward-epsilon must be positive.")
    if args.reward_instability_lambda < 0.0:
        raise ValueError("--reward-instability-lambda must be non-negative.")
    if args.reward_clip_min > args.reward_clip_max:
        raise ValueError("--reward-clip-min must be <= --reward-clip-max.")
    if args.context_trend_window <= 0:
        raise ValueError("--context-trend-window must be positive.")
    if args.context_trend_epsilon < 0.0:
        raise ValueError("--context-trend-epsilon must be non-negative.")
    if args.scheduler_t_max is not None and args.scheduler_t_max <= 0:
        raise ValueError("--scheduler-t-max must be positive when provided.")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative.")
    if not 0.0 <= args.label_noise_rate <= 1.0:
        raise ValueError("--label-noise-rate must be in [0, 1].")
    if args.label_noise_type == "none" and args.label_noise_rate != 0.0:
        raise ValueError(
            "--label-noise-rate must be 0.0 when --label-noise-type is none.")

    lr_candidates = (
        parse_lr_candidates(args.lr_candidates)
        if args.lr_candidates is not None
        else [1.0]
    )
    noise_candidates = (
        parse_noise_candidates(args.noise_candidates)
        if args.noise_candidates is not None
        else [0.0]
    )

    output = args.output or build_default_output_path(
        args.control_mode,
        args.optimizer,
        args.seed,
    )
    return ExperimentConfig(
        control_mode=args.control_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        momentum=args.momentum,
        lr_scheduler=args.lr_scheduler,
        scheduler_t_max=args.scheduler_t_max,
        warmup_epochs=args.warmup_epochs,
        reward_epsilon=args.reward_epsilon,
        reward_instability_lambda=args.reward_instability_lambda,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        episode_length=args.episode_length,
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
        structured_control_mode=args.structured_control_mode,
        context_mode=args.context_mode,
        context_trend_window=args.context_trend_window,
        context_trend_epsilon=args.context_trend_epsilon,
        label_noise_type=args.label_noise_type,
        label_noise_rate=args.label_noise_rate,
        seed=args.seed,
        output=output,
        ema_alpha=float(DEFAULT_CONFIG["ema_alpha"]),
        num_workers=int(DEFAULT_CONFIG["num_workers"]),
        data_dir=str(DEFAULT_CONFIG["data_dir"]),
        run_tag=args.run_tag,
    )


def build_default_output_path(
    control_mode: ControlMode,
    optimizer: OptimizerName,
    seed: int,
) -> str:
    """Build the default JSON output path."""

    return str(
        PROJECT_ROOT
        / "results"
        / f"cifar100_{control_mode}_{optimizer.lower()}_seed{seed}.json"
    )


def control_mode_name(control_mode: ControlMode) -> str:
    """Return the stable result-facing name for a control regime."""

    return CONTROL_MODE_NAMES[control_mode]


def build_experiment_label(config: ExperimentConfig) -> str:
    """Return a compact label combining control regime and optimizer family."""

    return f"{control_mode_name(config.control_mode)}+{config.optimizer}"


def set_seed(seed: int) -> None:
    """Seed Python and torch for reproducible runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def import_torchvision_modules() -> tuple[object, object, object]:
    """Import torchvision lazily so the script remains importable without it."""

    try:
        from torchvision import datasets, models, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchvision is required for CIFAR-100 experiments."
        ) from exc
    return datasets, models, transforms


def build_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, LabelNoiseMetadata | None]:
    """Create CIFAR-100 loaders and optional training-label noise metadata."""

    datasets, _, transforms = import_torchvision_modules()
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    data_root = PROJECT_ROOT / config.data_dir
    train_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=True,
    )
    test_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=False,
        transform=test_transform,
        download=True,
    )
    label_noise_metadata = apply_label_noise(train_dataset, config)
    train_eval_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        transform=test_transform,
        download=True,
    )
    train_eval_dataset.targets = list(train_dataset.targets)
    indexed_train_dataset = IndexedDataset(train_dataset)
    indexed_train_eval_dataset = IndexedDataset(train_eval_dataset)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        indexed_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = DataLoader(
        indexed_train_eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, train_eval_loader, label_noise_metadata


def apply_label_noise(
    train_dataset: "CIFAR100",
    config: ExperimentConfig,
) -> LabelNoiseMetadata | None:
    """Dispatch reproducible training-label corruption and return provenance."""

    targets = getattr(train_dataset, "targets", None)
    if not isinstance(targets, list):
        raise TypeError("Expected CIFAR100.targets to be a mutable list.")
    original_targets = [int(target) for target in targets]
    if config.label_noise_type == "none" or config.label_noise_rate <= 0.0:
        return None
    num_examples = len(targets)
    num_corrupt = int(round(num_examples * config.label_noise_rate))
    if num_corrupt <= 0:
        return None
    rng = random.Random(config.seed)
    indices = list(range(num_examples))
    rng.shuffle(indices)
    selected_indices = indices[:num_corrupt]
    corrupted_targets = list(original_targets)
    if config.label_noise_type == "symmetric":
        apply_symmetric_label_noise(corrupted_targets, selected_indices, rng)
    elif config.label_noise_type == "asymmetric":
        apply_asymmetric_label_noise(corrupted_targets, selected_indices)
    else:
        raise ValueError(
            f"Unsupported label noise type: {config.label_noise_type}")
    train_dataset.targets = corrupted_targets
    corrupted_index_set = set(selected_indices)
    corrupted_mask = [
        index in corrupted_index_set for index in range(num_examples)]
    clean_mask = [not is_corrupted for is_corrupted in corrupted_mask]
    return LabelNoiseMetadata(
        original_targets=original_targets,
        noisy_targets=[int(target) for target in corrupted_targets],
        corrupted_mask=corrupted_mask,
        clean_mask=clean_mask,
        corrupted_indices=list(selected_indices),
    )


def apply_symmetric_label_noise(
    targets: list[int],
    selected_indices: list[int],
    rng: random.Random,
) -> None:
    """Replace selected labels with uniformly sampled incorrect labels."""

    for index in selected_indices:
        original_label = int(targets[index])
        sampled = rng.randrange(99)
        targets[index] = sampled if sampled < original_label else sampled + 1


def apply_asymmetric_label_noise(
    targets: list[int],
    selected_indices: list[int],
) -> None:
    """Replace selected labels by cycling to the next fine label in each superclass."""

    for index in selected_indices:
        targets[index] = CIFAR100_ASYMMETRIC_LABEL_MAPPING[int(targets[index])]


def label_noise_enabled(config: ExperimentConfig) -> bool:
    """Return whether this run should produce noisy-label diagnostics."""

    return config.label_noise_type != "none" and config.label_noise_rate > 0.0


def build_model(device: torch.device) -> nn.Module:
    """Build a CIFAR-adapted ResNet-18."""

    _, models, _ = import_torchvision_modules()
    model = models.resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def build_context_bucket_names(context_mode: str) -> list[str] | None:
    """Return the bucket names for one shared context mode."""

    if context_mode == "none":
        return None
    if context_mode == "trend":
        return list(TREND_CONTEXT_BUCKETS)
    return list(TREND_PHASE_CONTEXT_BUCKETS)


def format_axis_value(value: float) -> str:
    """Format one axis value for concise initialization logs."""

    return repr(float(value))


def format_axis_values(values: list[float]) -> str:
    """Format one candidate list for concise initialization logs."""

    return "[" + ", ".join(format_axis_value(value) for value in values) + "]"


def print_structured_axis_summary(
    lr_candidates: list[float],
    noise_candidates: list[float],
) -> None:
    """Print a one-time summary of adaptive versus fixed structured axes."""

    adaptive_axes: list[str] = []
    fixed_axes: list[str] = []
    axis_summaries = (
        ("lr_multiplier", lr_candidates),
        ("noise_std", noise_candidates),
    )
    for axis_name, values in axis_summaries:
        if len(values) == 1:
            fixed_axes.append(f"{axis_name}={format_axis_value(values[0])}")
        else:
            adaptive_axes.append(f"{axis_name}={format_axis_values(values)}")
    print(f"adaptive axes: {', '.join(adaptive_axes) or '<none>'}")
    print(f"fixed axes: {', '.join(fixed_axes) or '<none>'}")


def build_axis_controller(
    *,
    arm_values: list[float],
    control_mode: ControlMode,
    random_seed: int,
    context_mode: str,
) -> object | None:
    """Build one LR or noise controller, or return None for fixed axes."""

    if len(arm_values) == 1:
        return None
    if control_mode == "random":
        return RandomController(n_arms=len(arm_values), random_seed=random_seed)
    if control_mode != "adaptive":
        raise ValueError(
            f"Unsupported structured control mode: {control_mode}")
    bucket_names = build_context_bucket_names(context_mode)
    if bucket_names is not None:
        return BucketedContextualController(
            n_arms=len(arm_values),
            bucket_names=bucket_names,
            random_seed=random_seed,
            prior_from_global=True,
        )
    return DiscountedUCBController(n_arms=len(arm_values), random_seed=random_seed)


def build_baseline_optimizer(
    config: ExperimentConfig,
    parameters: Any,
) -> torch.optim.Optimizer:
    """Build the selected base optimizer for non-episodic baseline runs."""

    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "SGD":
        return AdaptiveModeSGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            mode=CandidateConfig(name="base"),
            noise_seed=config.seed,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_wrapped_optimizer(
    config: ExperimentConfig,
    parameters: Any,
    mode: CandidateConfig,
) -> AdaptiveOptimizer:
    """Build the selected AEES-compatible optimizer wrapper."""

    if config.optimizer == "AdamW":
        return AdaptiveModeAdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            mode=mode,
            noise_seed=config.seed,
        )
    if config.optimizer == "SGD":
        return AdaptiveModeSGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            mode=mode,
            noise_seed=config.seed,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_method_components(
    config: ExperimentConfig,
    model: nn.Module,
    total_training_steps: int,
) -> TrainingComponents:
    """Build optimizer, controller, and episode manager objects for one run."""

    parameters = model.parameters()
    if config.control_mode == "baseline":
        return TrainingComponents(
            optimizer=build_baseline_optimizer(config, parameters),
            episode_manager=None,
            method_name=control_mode_name(config.control_mode),
            controller_logs=None,
        )

    reward_fn = NormalizedLossImprovementReward(
        reward_epsilon=config.reward_epsilon,
        reward_instability_lambda=config.reward_instability_lambda,
        reward_clip_min=config.reward_clip_min,
        reward_clip_max=config.reward_clip_max,
    )

    lr_candidates = list(config.lr_candidates)
    noise_candidates = list(config.noise_candidates)
    print_structured_axis_summary(lr_candidates, noise_candidates)
    initial_mode = CandidateConfig(
        name=build_candidate_configs_from_axes(
            [lr_candidates[0]],
            [noise_candidates[0]],
        )[0].name,
        lr_multiplier=lr_candidates[0],
        noise_std=noise_candidates[0],
    )
    optimizer = build_wrapped_optimizer(config, parameters, initial_mode)
    lr_controller = build_axis_controller(
        arm_values=lr_candidates,
        control_mode=config.control_mode,
        random_seed=config.seed,
        context_mode=config.context_mode,
    )
    if config.structured_control_mode == "independent":
        noise_controller: object | list[object] = build_axis_controller(
            arm_values=noise_candidates,
            control_mode=config.control_mode,
            random_seed=config.seed + 101,
            context_mode=config.context_mode,
        )
    elif len(lr_candidates) == 1:
        noise_controller = build_axis_controller(
            arm_values=noise_candidates,
            control_mode=config.control_mode,
            random_seed=config.seed + 101,
            context_mode=config.context_mode,
        )
    else:
        noise_controller = [
            build_axis_controller(
                arm_values=noise_candidates,
                control_mode=config.control_mode,
                random_seed=config.seed + 101 + lr_index,
                context_mode=config.context_mode,
            )
            for lr_index in range(len(lr_candidates))
        ]
    episode_manager = StructuredEpisodeManager(
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
        lr_controller=lr_controller,
        noise_controller=noise_controller,
        reward_fn=reward_fn,
        episode_length=config.episode_length,
        structured_control_mode=config.structured_control_mode,
        context_mode=config.context_mode,
        total_training_steps=total_training_steps if config.context_mode == "trend_phase" else None,
        context_trend_window=config.context_trend_window,
        context_trend_epsilon=config.context_trend_epsilon,
        ema_alpha=config.ema_alpha,
    )
    return TrainingComponents(
        optimizer=optimizer,
        episode_manager=episode_manager,
        method_name=control_mode_name(config.control_mode),
        controller_logs=build_structured_controller_logs_container(
            control_mode=config.control_mode,
            structured_control_mode=config.structured_control_mode,
            context_mode=config.context_mode,
            lr_controller=lr_controller,
            noise_controller=noise_controller,
            lr_candidates=lr_candidates,
            noise_candidates=noise_candidates,
        ),
        structured_control_mode=config.structured_control_mode,
        lr_candidates=lr_candidates,
        noise_candidates=noise_candidates,
        lr_controller=lr_controller,
        noise_controller=noise_controller,
    )


def build_lr_scheduler(
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the configured epoch-level scheduler."""

    if config.lr_scheduler == "none":
        return None
    schedule_span = resolve_scheduler_t_max(config)
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_span)
    if config.lr_scheduler == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=build_linear_lr_lambda(schedule_span),
        )
    if config.lr_scheduler == "warmup_linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=build_warmup_linear_lr_lambda(
                total_epochs=schedule_span,
                warmup_epochs=config.warmup_epochs,
            ),
        )
    raise ValueError(f"Unsupported lr scheduler: {config.lr_scheduler}")


def resolve_scheduler_t_max(config: ExperimentConfig) -> int:
    """Resolve the scheduler span used by cosine and linear schedules."""

    return config.scheduler_t_max or config.epochs


def build_linear_lr_lambda(total_epochs: int) -> Any:
    """Return a simple epoch-level linear decay schedule."""

    def lr_lambda(epoch_index: int) -> float:
        progress = min(max((epoch_index + 1) / max(total_epochs, 1), 0.0), 1.0)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def build_warmup_linear_lr_lambda(total_epochs: int, warmup_epochs: int) -> Any:
    """Return a simple epoch-level warmup-then-linear schedule."""

    warmup_epochs = min(max(warmup_epochs, 0), max(total_epochs - 1, 0))

    def lr_lambda(epoch_index: int) -> float:
        step_index = epoch_index + 1
        if warmup_epochs > 0 and step_index <= warmup_epochs:
            return min(step_index / warmup_epochs, 1.0)
        decay_epochs = max(total_epochs - warmup_epochs, 1)
        decay_progress = min(
            max((step_index - warmup_epochs) / decay_epochs, 0.0), 1.0)
        return max(0.0, 1.0 - decay_progress)

    return lr_lambda


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    components: TrainingComponents,
    global_step: int,
    flop_accumulator: FlopAccumulator,
) -> tuple[float, float, int]:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        inputs, targets, _ = unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if components.episode_manager is not None:
            mode = components.episode_manager.on_step_start(global_step)
            adaptive_optimizer = components.optimizer
            if not isinstance(
                adaptive_optimizer,
                (AdaptiveModeAdamW, AdaptiveModeSGD),
            ):
                raise TypeError(
                    "Episode-managed methods require an adaptive optimizer wrapper."
                )
            adaptive_optimizer.set_mode(mode)

        components.optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        components.optimizer.step()
        flop_accumulator.add_train_batch(
            estimate_resnet18_cifar100_train_flops_per_batch(
                int(targets.size(0)))
        )

        predictions = logits.argmax(dim=1)
        batch_size = int(targets.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        total_correct += int((predictions == targets).sum().item())

        if components.episode_manager is not None:
            update_norm = (
                components.optimizer.compute_update_norm()
                if isinstance(
                    components.optimizer,
                    (AdaptiveModeAdamW, AdaptiveModeSGD),
                )
                else None
            )
            components.episode_manager.on_step_end(
                float(loss.item()), update_norm=update_norm)
            collect_controller_snapshot_if_episode_closed(components)
        global_step += 1

    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
        global_step,
    )


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    flop_accumulator: FlopAccumulator,
    include_mask: list[bool] | None = None,
    target_override: list[int] | None = None,
) -> float:
    """Evaluate top-1 accuracy, optionally on an indexed subset and targets."""

    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        inputs, targets, indices = unpack_batch(batch)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        flop_accumulator.add_eval_batch(
            estimate_resnet18_cifar100_eval_flops_per_batch(
                int(targets.size(0)))
        )
        predictions = logits.argmax(dim=1)
        if indices is not None:
            selected = build_subset_selection(indices, include_mask, device)
            if target_override is not None:
                targets = build_target_override(
                    indices, target_override, device)
        else:
            if include_mask is not None or target_override is not None:
                raise ValueError("Subset evaluation requires indexed batches.")
            selected = torch.ones_like(targets, dtype=torch.bool)
        if int(selected.sum().item()) == 0:
            continue
        correct += int((predictions[selected] ==
                       targets[selected]).sum().item())
        total += int(selected.sum().item())
    return correct / max(total, 1)


def unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return inputs, targets, and optional original dataset indices."""

    if len(batch) == 3:
        inputs, targets, indices = batch
        return inputs, targets, indices
    if len(batch) == 2:
        inputs, targets = batch
        return inputs, targets, None
    raise ValueError(f"Expected a 2- or 3-item batch, got {len(batch)} items.")


def build_subset_selection(
    indices: torch.Tensor,
    include_mask: list[bool] | None,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean selection tensor for indexed subset evaluation."""

    if include_mask is None:
        return torch.ones(indices.size(0), dtype=torch.bool, device=device)
    selected = [bool(include_mask[int(index)]) for index in indices.tolist()]
    return torch.tensor(selected, dtype=torch.bool, device=device)


def build_target_override(
    indices: torch.Tensor,
    target_override: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build alternate targets for indexed evaluation."""

    targets = [int(target_override[int(index)]) for index in indices.tolist()]
    return torch.tensor(targets, dtype=torch.long, device=device)


def build_label_noise_diagnostics(
    *,
    config: ExperimentConfig,
    metadata: LabelNoiseMetadata | None,
    model: nn.Module,
    train_eval_loader: DataLoader,
    device: torch.device,
    final_val_accuracy: float,
) -> dict[str, object] | None:
    """Compute final train-subset diagnostics for noisy-label runs."""

    if metadata is None:
        return None
    diagnostic_flops = FlopAccumulator()
    train_all_accuracy = evaluate_accuracy(
        model,
        train_eval_loader,
        device,
        diagnostic_flops,
    )
    train_clean_accuracy = evaluate_accuracy(
        model,
        train_eval_loader,
        device,
        diagnostic_flops,
        include_mask=metadata.clean_mask,
    )
    train_corrupted_accuracy_vs_noisy_labels = evaluate_accuracy(
        model,
        train_eval_loader,
        device,
        diagnostic_flops,
        include_mask=metadata.corrupted_mask,
    )
    train_corrupted_accuracy_vs_clean_labels = evaluate_accuracy(
        model,
        train_eval_loader,
        device,
        diagnostic_flops,
        include_mask=metadata.corrupted_mask,
        target_override=metadata.original_targets,
    )
    num_train_examples = len(metadata.original_targets)
    num_corrupted_examples = len(metadata.corrupted_indices)
    num_clean_examples = num_train_examples - num_corrupted_examples
    return {
        "label_noise": {
            "enabled": True,
            "type": config.label_noise_type,
            "rate": config.label_noise_rate,
            "num_train_examples": num_train_examples,
            "num_clean_examples": num_clean_examples,
            "num_corrupted_examples": num_corrupted_examples,
            "corrupted_fraction": (
                num_corrupted_examples / max(num_train_examples, 1)
            ),
            "original_targets": metadata.original_targets,
            "noisy_targets": metadata.noisy_targets,
            "corrupted_mask": metadata.corrupted_mask,
            "clean_mask": metadata.clean_mask,
            "corrupted_indices": metadata.corrupted_indices,
        },
        "final_subset_accuracies": {
            "train_all_accuracy": train_all_accuracy,
            "train_clean_accuracy": train_clean_accuracy,
            "train_corrupted_accuracy_vs_noisy_labels": (
                train_corrupted_accuracy_vs_noisy_labels
            ),
            "train_corrupted_accuracy_vs_clean_labels": (
                train_corrupted_accuracy_vs_clean_labels
            ),
            "final_val_accuracy": final_val_accuracy,
        },
        "label_noise_diagnostic_eval_flops": diagnostic_flops.total_flops,
    }


def build_result(
    config: ExperimentConfig,
    components: TrainingComponents,
    train_losses: list[float],
    train_accuracies: list[float] | None,
    val_accuracies: list[float],
    epoch_wall_clock_seconds: list[float],
    epoch_tflops: list[float] | None,
    best_val_accuracy: float,
    final_val_accuracy: float,
    final_train_loss: float,
    total_steps: int,
    wall_clock_seconds: float,
    estimated_flops: float | None,
    accuracy_per_tflop: float | None,
    label_noise_diagnostics: dict[str, object] | None = None,
) -> RunResult:
    """Create the final JSON-serializable run summary."""

    episode_logs = None
    if components.episode_manager is not None:
        episode_logs = build_episode_logs(
            components.episode_manager.get_logs())

    config_dict = asdict(config)
    config_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["pin_memory"] = torch.cuda.is_available()
    config_dict["control_mode"] = config.control_mode
    config_dict["control_regime"] = control_mode_name(config.control_mode)
    config_dict["experiment_label"] = build_experiment_label(config)
    config_dict["base_lr"] = config.lr
    config_dict["base_weight_decay"] = config.weight_decay
    config_dict["optimizer_family"] = config.optimizer
    config_dict["optimizer_momentum"] = config.momentum
    config_dict["lr_scheduler"] = config.lr_scheduler
    config_dict["scheduler_t_max"] = resolve_scheduler_t_max(config)
    config_dict["scheduler_active"] = config.lr_scheduler != "none"
    config_dict["warmup_epochs"] = config.warmup_epochs
    if label_noise_diagnostics is not None:
        diagnostics = config_dict.setdefault("diagnostics", {})
        if not isinstance(diagnostics, dict):
            raise TypeError("Expected config diagnostics to be a dictionary.")
        diagnostics.update(label_noise_diagnostics)
    if label_noise_enabled(config):
        diagnostics = config_dict.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise RuntimeError(
                "Noisy-label run did not attach config['diagnostics']."
            )
        for required_key in ["label_noise", "final_subset_accuracies"]:
            if required_key not in diagnostics:
                raise RuntimeError(
                    "Noisy-label run did not attach "
                    f"config['diagnostics']['{required_key}']."
                )
    if components.episode_manager is not None:
        lr_candidates = list(components.lr_candidates or [])
        noise_candidates = list(components.noise_candidates or [])
        candidate_configs = build_candidate_configs_from_axes(
            lr_candidates, noise_candidates)
        config_dict["lr_candidates"] = lr_candidates
        config_dict["noise_candidates"] = noise_candidates
        config_dict["structured_control_mode"] = config.structured_control_mode
        config_dict["context_mode"] = config.context_mode
        config_dict["context_trend_window"] = config.context_trend_window
        config_dict["context_trend_epsilon"] = config.context_trend_epsilon
        config_dict["candidate_config_names"] = [
            candidate.name for candidate in candidate_configs]
        config_dict["candidate_config_definitions"] = [
            {
                "name": candidate.name,
                "lr_multiplier": candidate.lr_multiplier,
                "noise_std": candidate.noise_std,
            }
            for candidate in candidate_configs
        ]

    return RunResult(
        method_name=components.method_name,
        task_name=config.task_name,
        seed=config.seed,
        model_name=config.model_name,
        dataset_name="CIFAR-100",
        config=config_dict,
        best_val_accuracy=best_val_accuracy,
        final_val_accuracy=final_val_accuracy,
        final_train_loss=final_train_loss,
        total_steps=total_steps,
        total_epochs=config.epochs,
        wall_clock_seconds=wall_clock_seconds,
        estimated_flops=estimated_flops,
        accuracy_per_tflop=accuracy_per_tflop,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        epoch_wall_clock_seconds=epoch_wall_clock_seconds,
        epoch_tflops=epoch_tflops,
        episode_logs=episode_logs,
        controller_logs=components.controller_logs,
        run_tag=config.run_tag,
    )


def run_experiment(config: ExperimentConfig) -> RunResult:
    """Run one full CIFAR-100 experiment."""

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_eval_loader, label_noise_metadata = build_dataloaders(
        config
    )
    if label_noise_enabled(config) and label_noise_metadata is None:
        raise RuntimeError(
            "Label-noise metadata was not created for an enabled noisy-label run."
        )
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    components = build_method_components(
        config,
        model,
        total_training_steps=len(train_loader) * config.epochs,
    )
    lr_scheduler = build_lr_scheduler(config, components.optimizer)

    start_time = time.perf_counter()
    global_step = 0
    best_val_accuracy = 0.0
    final_val_accuracy = 0.0
    final_train_loss = 0.0
    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    epoch_wall_clock_seconds: list[float] = []
    epoch_tflops: list[float] = []
    flop_accumulator = FlopAccumulator()

    for epoch_index in range(config.epochs):
        epoch_start_time = time.perf_counter()
        epoch_episode_start = get_completed_episode_count(components)
        final_train_loss, train_accuracy, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            components=components,
            global_step=global_step,
            flop_accumulator=flop_accumulator,
        )
        final_val_accuracy = evaluate_accuracy(
            model, test_loader, device, flop_accumulator)
        best_val_accuracy = max(best_val_accuracy, final_val_accuracy)
        train_losses.append(final_train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(final_val_accuracy)
        epoch_wall_clock_seconds.append(time.perf_counter() - epoch_start_time)
        epoch_tflops.append(flop_accumulator.finish_epoch())
        print_epoch_status(
            method_name=build_experiment_label(config),
            epoch_index=epoch_index,
            total_epochs=config.epochs,
            train_loss=final_train_loss,
            train_accuracy=train_accuracy,
            val_accuracy=final_val_accuracy,
            epoch_seconds=epoch_wall_clock_seconds[-1],
            cumulative_tflops=flop_accumulator.total_tflops,
            episode_manager=components.episode_manager,
            epoch_episode_start=epoch_episode_start,
        )
        if lr_scheduler is not None:
            lr_scheduler.step()

    if components.episode_manager is not None:
        previous_episode_count = len(
            components.episode_manager.get_logs()["episode_rewards"])
        components.episode_manager.finalize()
        collect_controller_snapshot_if_episode_closed(
            components,
            previous_episode_count=previous_episode_count,
        )

    estimated_flops = flop_accumulator.total_flops
    accuracy_per_tflop = None
    if flop_accumulator.total_tflops > 0.0:
        accuracy_per_tflop = best_val_accuracy / flop_accumulator.total_tflops
    label_noise_diagnostics = build_label_noise_diagnostics(
        config=config,
        metadata=label_noise_metadata,
        model=model,
        train_eval_loader=train_eval_loader,
        device=device,
        final_val_accuracy=final_val_accuracy,
    )
    if label_noise_enabled(config) and label_noise_diagnostics is None:
        raise RuntimeError(
            "Final label-noise diagnostics were not computed for a noisy-label run."
        )
    return build_result(
        config=config,
        components=components,
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        epoch_wall_clock_seconds=epoch_wall_clock_seconds,
        epoch_tflops=epoch_tflops,
        best_val_accuracy=best_val_accuracy,
        final_val_accuracy=final_val_accuracy,
        final_train_loss=final_train_loss,
        total_steps=global_step,
        wall_clock_seconds=time.perf_counter() - start_time,
        estimated_flops=estimated_flops,
        accuracy_per_tflop=accuracy_per_tflop,
        label_noise_diagnostics=label_noise_diagnostics,
    )


def build_structured_controller_logs_container(
    *,
    control_mode: ControlMode,
    structured_control_mode: str,
    context_mode: str,
    lr_controller: object | None,
    noise_controller: object | list[object] | None,
    lr_candidates: list[float],
    noise_candidates: list[float],
) -> dict[str, object]:
    """Create controller log storage for structured AEES runs."""

    logs: dict[str, object] = {
        "structured_control_mode": structured_control_mode,
        "context_mode": context_mode,
        "_logged_episode_count": 0,
        "lr_controller_logs": build_axis_controller_logs_container(
            controller=lr_controller,
            arm_values=lr_candidates,
            axis_name="lr",
        ),
    }
    if not isinstance(noise_controller, list):
        logs["noise_controller_logs"] = build_axis_controller_logs_container(
            controller=noise_controller,
            arm_values=noise_candidates,
            axis_name="noise",
        )
    else:
        noise_logs_by_lr: dict[str, object] = {}
        for lr_value, controller in zip(lr_candidates, noise_controller, strict=True):
            controller_log = build_axis_controller_logs_container(
                controller=controller,
                arm_values=noise_candidates,
                axis_name="noise",
            )
            controller_log["conditioned_on_lr_value"] = safe_float(lr_value)
            noise_logs_by_lr[str(lr_value)] = controller_log
        logs["noise_controller_logs_by_lr"] = noise_logs_by_lr
    if control_mode == "random":
        logs["non_contextual_random_note"] = (
            "RandomScheduler keeps random controllers non-contextual; context is logged per episode only."
        )
    return logs


def build_axis_controller_logs_container(
    *,
    controller: object | None,
    arm_values: list[float],
    axis_name: str,
) -> dict[str, object]:
    """Create a logging container for one LR or noise controller."""

    logs: dict[str, object] = {
        "controller_type": "FixedAxis" if controller is None else type(controller).__name__,
        "axis_name": axis_name,
        "arm_values": [safe_float(value) for value in arm_values],
        "controller_created": controller is not None,
    }
    if controller is None:
        logs["fixed_arm_value"] = safe_float(arm_values[0])
        return logs
    if isinstance(controller, BucketedContextualController):
        logs["context_bucket_names"] = list(controller.bucket_names)
        logs["prior_from_global"] = controller.prior_from_global
        logs["bucket_visit_counts_history"] = []
        logs["bucket_reward_sums_history"] = []
        logs["bucket_distinct_arm_counts_history"] = []
        logs["global_prior_logs"] = {
            "value_estimates_history": [],
            "effective_counts_history": [],
            "warmup_counts_history": [],
            "controller_updates_history": [],
        }
        logs["bucket_controller_logs"] = {
            bucket_name: {
                "bucket_id": bucket_name,
                "value_estimates_history": [],
                "effective_counts_history": [],
                "warmup_counts_history": [],
                "controller_updates_history": [],
                "bucket_visit_count_history": [],
            }
            for bucket_name in controller.bucket_names
        }
        return logs
    initialize_discounted_ucb_histories(logs, controller)
    return logs


def initialize_discounted_ucb_histories(logs: dict[str, object], controller: object) -> None:
    """Add adaptive-controller history slots when the controller uses discounted UCB."""

    if isinstance(controller, DiscountedUCBController):
        logs["value_estimates_history"] = []
        logs["effective_counts_history"] = []
        logs["warmup_counts_history"] = []
        logs["controller_updates_history"] = []


def collect_controller_snapshot_if_episode_closed(
    components: TrainingComponents,
    previous_episode_count: int | None = None,
) -> None:
    """Append controller-state snapshots when a new episode has just closed."""

    if components.episode_manager is None or components.controller_logs is None:
        return
    current_episode_count = len(
        components.episode_manager.get_logs()["episode_rewards"])
    if previous_episode_count is None:
        previous_episode_count = get_logged_controller_episode_count(
            components.controller_logs)
    if current_episode_count <= previous_episode_count:
        return

    for _ in range(current_episode_count - previous_episode_count):
        append_structured_controller_snapshot(components)
    components.controller_logs["_logged_episode_count"] = current_episode_count


def append_structured_controller_snapshot(components: TrainingComponents) -> None:
    """Store one snapshot of the structured controllers."""

    controller_logs = components.controller_logs
    if controller_logs is None:
        return
    lr_controller_logs = controller_logs.get("lr_controller_logs")
    if isinstance(lr_controller_logs, dict) and components.lr_controller is not None:
        append_controller_snapshot(
            components.lr_controller, lr_controller_logs)

    noise_logs = controller_logs.get("noise_controller_logs")
    if isinstance(noise_logs, dict):
        if components.noise_controller is not None and not isinstance(components.noise_controller, list):
            append_controller_snapshot(components.noise_controller, noise_logs)
        return

    noise_logs_by_lr = controller_logs.get("noise_controller_logs_by_lr")
    if not isinstance(noise_logs_by_lr, dict):
        return
    if not isinstance(components.noise_controller, list) or components.lr_candidates is None:
        return
    for lr_value, controller in zip(components.lr_candidates, components.noise_controller, strict=True):
        controller_log = noise_logs_by_lr.get(str(lr_value))
        if isinstance(controller_log, dict):
            append_controller_snapshot(controller, controller_log)


def append_controller_snapshot(controller: object, controller_logs: dict[str, object]) -> None:
    """Store one snapshot for a structured controller."""

    if isinstance(controller, BucketedContextualController):
        append_contextual_controller_snapshot(controller, controller_logs)
    elif isinstance(controller, DiscountedUCBController):
        labels = [str(value)
                  for value in controller_logs.get("arm_values", [])]
        append_discounted_ucb_state_snapshot(
            state=controller.get_state(),
            labels=labels,
            controller_logs=controller_logs,
        )


def append_contextual_controller_snapshot(
    controller: BucketedContextualController,
    controller_logs: dict[str, object],
) -> None:
    """Store one contextual-controller snapshot including bucket summaries."""

    state = controller.get_state()
    labels = [str(value) for value in controller_logs.get("arm_values", [])]
    bucket_visit_counts = state.get("bucket_visit_counts", {})
    bucket_reward_sums = state.get("bucket_reward_sums", {})
    bucket_distinct_arm_counts = state.get("bucket_distinct_arm_counts", {})
    bucket_states = state.get("bucket_states", {})
    global_prior_state = state.get("global_prior_state", {})

    visit_history = controller_logs.setdefault(
        "bucket_visit_counts_history", [])
    reward_history = controller_logs.setdefault(
        "bucket_reward_sums_history", [])
    distinct_history = controller_logs.setdefault(
        "bucket_distinct_arm_counts_history", [])
    if isinstance(visit_history, list):
        visit_history.append({key: int(value)
                             for key, value in dict(bucket_visit_counts).items()})
    if isinstance(reward_history, list):
        reward_history.append({key: safe_float(value)
                              for key, value in dict(bucket_reward_sums).items()})
    if isinstance(distinct_history, list):
        distinct_history.append(
            {key: int(value) for key, value in dict(bucket_distinct_arm_counts).items()})

    global_prior_logs = controller_logs.get("global_prior_logs")
    if isinstance(global_prior_logs, dict) and isinstance(global_prior_state, dict):
        append_discounted_ucb_state_snapshot(
            state=global_prior_state,
            labels=labels,
            controller_logs=global_prior_logs,
        )

    bucket_controller_logs = controller_logs.get("bucket_controller_logs")
    if not isinstance(bucket_controller_logs, dict):
        return
    for bucket_name, bucket_log in bucket_controller_logs.items():
        if not isinstance(bucket_log, dict):
            continue
        visit_count_history = bucket_log.setdefault(
            "bucket_visit_count_history", [])
        if isinstance(visit_count_history, list):
            visit_count_history.append(
                int(dict(bucket_visit_counts).get(bucket_name, 0)))
        bucket_state = dict(bucket_states).get(bucket_name)
        if bucket_state is None:
            for key in [
                "value_estimates_history",
                "effective_counts_history",
                "warmup_counts_history",
                "controller_updates_history",
            ]:
                history = bucket_log.setdefault(key, [])
                if isinstance(history, list):
                    history.append(None)
            continue
        if isinstance(bucket_state, dict):
            append_discounted_ucb_state_snapshot(
                state=bucket_state,
                labels=labels,
                controller_logs=bucket_log,
            )


def append_discounted_ucb_state_snapshot(
    *,
    state: dict[str, object],
    labels: list[str],
    controller_logs: dict[str, object],
) -> None:
    """Append one discounted-UCB state snapshot into the provided log container."""

    mean_rewards = list(state.get("mean_rewards", []))
    counts = list(state.get("counts", []))
    warmup_counts = list(state.get("warmup_counts", []))
    value_estimates = {
        label: safe_float(mean_reward)
        for label, mean_reward in zip(labels, mean_rewards, strict=False)
    }
    effective_counts = {
        label: safe_float(count)
        for label, count in zip(labels, counts, strict=False)
    }
    warmup_snapshot = {
        label: int(count)
        for label, count in zip(labels, warmup_counts, strict=False)
    }

    for key, value in [
        ("value_estimates_history", value_estimates),
        ("effective_counts_history", effective_counts),
        ("warmup_counts_history", warmup_snapshot),
        ("controller_updates_history", int(state.get("total_updates", 0))),
    ]:
        history = controller_logs.setdefault(key, [])
        if isinstance(history, list):
            history.append(value)


def get_logged_controller_episode_count(controller_logs: dict[str, object]) -> int:
    """Return how many episodes already have controller snapshots."""

    logged_episode_count = controller_logs.get("_logged_episode_count")
    return int(logged_episode_count) if isinstance(logged_episode_count, int) else 0


def build_episode_logs(raw_logs: dict[str, list[object]]) -> dict[str, object]:
    """Return the episode logs in a JSON-friendly form."""

    normalized = {key: list(values) for key, values in raw_logs.items()}
    if "selected_combined_names" in normalized:
        normalized["selected_candidate_names"] = list(
            normalized["selected_combined_names"])
    return normalized


def print_epoch_status(
    *,
    method_name: str,
    epoch_index: int,
    total_epochs: int,
    train_loss: float,
    train_accuracy: float | None,
    val_accuracy: float,
    epoch_seconds: float,
    cumulative_tflops: float | None,
    episode_manager: StructuredEpisodeManager | None,
    epoch_episode_start: int,
) -> None:
    """Print one concise epoch status line."""

    parts = [
        f"[{method_name}]",
        f"epoch {epoch_index + 1}/{total_epochs}",
        f"train_loss={train_loss:.4f}",
    ]
    if train_accuracy is not None:
        parts.append(f"train_acc={train_accuracy:.4f}")
    parts.append(f"val_acc={val_accuracy:.4f}")
    parts.append(f"time={epoch_seconds:.1f}s")
    if cumulative_tflops is not None:
        parts.append(f"cum_tflops={cumulative_tflops:.3f}")

    scheduler_summary = build_scheduler_epoch_summary(
        episode_manager=episode_manager,
        epoch_episode_start=epoch_episode_start,
    )
    if scheduler_summary is not None:
        parts.append(scheduler_summary)
    print(" | ".join(parts))


def build_scheduler_epoch_summary(
    *,
    episode_manager: StructuredEpisodeManager | None,
    epoch_episode_start: int,
) -> str | None:
    """Build a short per-epoch summary for scheduler methods."""

    if episode_manager is None:
        return None
    raw_logs = episode_manager.get_logs()
    total_episodes = len(raw_logs["episode_rewards"])
    selected_names = raw_logs.get("selected_combined_names", [])
    last_mode = str(selected_names[-1]) if selected_names else "-"
    epoch_rewards = [
        float(reward)
        for reward in raw_logs["episode_rewards"][epoch_episode_start:total_episodes]
        if reward is not None
    ]
    mean_reward_text = "n/a"
    if epoch_rewards:
        mean_reward_text = f"{(sum(epoch_rewards) / len(epoch_rewards)):.4f}"
    return (
        f"episodes={total_episodes}"
        f" last_mode={last_mode}"
        f" epoch_mean_reward={mean_reward_text}"
    )


def get_completed_episode_count(components: TrainingComponents) -> int:
    """Return the number of completed episodes logged so far."""

    if components.episode_manager is None:
        return 0
    return len(components.episode_manager.get_logs()["episode_rewards"])


def safe_float(value: object) -> float:
    """Convert one numeric value to a JSON-safe finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Expected a numeric value, got {type(value).__name__}.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("Expected a finite numeric value.")
    return converted


def main() -> None:
    """Run one experiment from the CLI and save the JSON result."""

    config = parse_args()
    result = run_experiment(config)
    save_run_result(result, config.output)
    print(f"Saved result to {config.output}")


if __name__ == "__main__":
    main()
