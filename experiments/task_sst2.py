"""Run an SST-2 experiment with AdamW or AEES structured scheduling."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import sys
import time
from typing import Any
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

# Stable local HF cache inside the repo/workspace
DEFAULT_HF_CACHE = PROJECT_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_CACHE / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(
    DEFAULT_HF_CACHE / "transformers"))

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
from pulseopt.optimizer import AdaptiveModeAdamW  # noqa: E402
from pulseopt.reward import NormalizedLossImprovementReward  # noqa: E402
from pulseopt.types import CandidateConfig  # noqa: E402
from experiments.utils.flops import (  # noqa: E402
    FlopAccumulator,
    estimate_distilbert_sst2_eval_flops_per_batch,
    estimate_distilbert_sst2_train_flops_per_batch,
)
from experiments.utils.metrics import RunResult  # noqa: E402
from experiments.utils.results import save_run_result  # noqa: E402


DEFAULT_CONFIG: dict[str, object] = {
    "task_name": "sst2",
    "dataset_name": "GLUE SST-2",
    "model_name": "distilbert-base-uncased",
    "epochs": 3,
    "batch_size": 16,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "lr_scheduler": "none",
    "scheduler_t_max": None,
    "warmup_epochs": 0,
    "reward_epsilon": 1e-8,
    "reward_instability_lambda": 0.0,
    "reward_clip_min": -1.0,
    "reward_clip_max": 1.0,
    "episode_length": 100,
    "ema_alpha": 0.1,
    "lr_candidates": [1.0],
    "noise_candidates": [0.0],
    "structured_control_mode": "independent",
    "context_mode": "none",
    "context_trend_window": 3,
    "context_trend_epsilon": 1e-3,
    "max_length": 128,
    "num_workers": 0,
    "cache_dir": str(DEFAULT_HF_CACHE),
    "local_files_only": False,
    "hf_token": os.environ.get("HF_TOKEN"),
    "tokenized_dataset_dir": str(PROJECT_ROOT / "data" / "sst2_tokenized"),
    "pretokenize_only": False,
    "label_noise_rate": 0.0,
    "label_noise_seed": 42,
    "run_tag": None,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed configuration for one SST-2 run."""

    method: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
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
    max_length: int
    seed: int
    output: str
    ema_alpha: float
    num_workers: int
    cache_dir: str | None
    local_files_only: bool
    hf_token: str | None
    tokenized_dataset_dir: str | None
    pretokenize_only: bool
    model_name: str = "distilbert-base-uncased"
    task_name: str = "sst2"
    dataset_name: str = "GLUE SST-2"
    label_noise_rate: float = 0.0
    label_noise_seed: int = 42
    run_tag: str | None = None


@dataclass
class TrainingComponents:
    """Method-specific objects used during training."""

    optimizer: torch.optim.Optimizer
    episode_manager: StructuredEpisodeManager | None
    method_name: str
    controller_logs: dict[str, object] | None
    structured_control_mode: str | None = None
    lr_candidates: list[float] | None = None
    noise_candidates: list[float] | None = None
    lr_controller: object | None = None
    noise_controller: object | list[object] | None = None


def parse_args() -> ExperimentConfig:
    """Parse the CLI into a typed experiment config."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["AdamW", "AdaptiveScheduler", "RandomScheduler"],
        default="AdamW",
    )
    parser.add_argument("--epochs", type=int,
                        default=int(DEFAULT_CONFIG["epochs"]))
    parser.add_argument("--batch-size", type=int,
                        default=int(DEFAULT_CONFIG["batch_size"]))
    parser.add_argument("--lr", type=float,
                        default=float(DEFAULT_CONFIG["lr"]))
    parser.add_argument("--weight-decay", type=float,
                        default=float(DEFAULT_CONFIG["weight_decay"]))
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "linear", "warmup_linear"],
        default=str(DEFAULT_CONFIG["lr_scheduler"]),
    )
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int,
                        default=int(DEFAULT_CONFIG["warmup_epochs"]))
    parser.add_argument("--reward-epsilon", type=float,
                        default=float(DEFAULT_CONFIG["reward_epsilon"]))
    parser.add_argument(
        "--reward-instability-lambda",
        type=float,
        default=float(DEFAULT_CONFIG["reward_instability_lambda"]),
    )
    parser.add_argument("--reward-clip-min", type=float,
                        default=float(DEFAULT_CONFIG["reward_clip_min"]))
    parser.add_argument("--reward-clip-max", type=float,
                        default=float(DEFAULT_CONFIG["reward_clip_max"]))
    parser.add_argument("--episode-length", type=int,
                        default=int(DEFAULT_CONFIG["episode_length"]))
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
    parser.add_argument("--context-trend-window", type=int,
                        default=int(DEFAULT_CONFIG["context_trend_window"]))
    parser.add_argument("--context-trend-epsilon", type=float,
                        default=float(DEFAULT_CONFIG["context_trend_epsilon"]))
    parser.add_argument("--max-length", type=int,
                        default=int(DEFAULT_CONFIG["max_length"]))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str,
                        default=str(DEFAULT_CONFIG["cache_dir"]))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-token", type=str,
                        default=DEFAULT_CONFIG["hf_token"])
    parser.add_argument("--run-tag", type=str, default=DEFAULT_CONFIG["run_tag"])
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Defaults to results/sst2_<method>_seed<seed>.json",
    )
    parser.add_argument(
        "--tokenized-dataset-dir",
        type=str,
        default=str(DEFAULT_CONFIG["tokenized_dataset_dir"]),
    )
    parser.add_argument(
        "--pretokenize-only",
        action="store_true",
        help="Only build and save the tokenized SST-2 dataset, then exit.",
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
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive.")

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

    output = args.output or build_default_output_path(args.method, args.seed)
    return ExperimentConfig(
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
        max_length=args.max_length,
        seed=args.seed,
        output=output,
        ema_alpha=float(DEFAULT_CONFIG["ema_alpha"]),
        num_workers=int(DEFAULT_CONFIG["num_workers"]),
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        hf_token=args.hf_token,
        tokenized_dataset_dir=args.tokenized_dataset_dir,
        pretokenize_only=bool(args.pretokenize_only),
        label_noise_rate=float(DEFAULT_CONFIG["label_noise_rate"]),
        label_noise_seed=int(DEFAULT_CONFIG["label_noise_seed"]),
        run_tag=args.run_tag,
    )


def build_default_output_path(method: str, seed: int) -> str:
    """Build the default JSON output path."""
    return str(PROJECT_ROOT / "results" / f"sst2_{method.lower()}_seed{seed}.json")


def set_seed(seed: int) -> None:
    """Seed Python and torch for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def import_sst2_modules() -> tuple[object, object, object, object, object]:
    """Import datasets and transformers lazily."""
    try:
        from datasets import load_dataset, load_from_disk
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "datasets and transformers are required for SST-2 experiments.") from exc
    return (
        load_dataset,
        load_from_disk,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
    )


def retry_with_backoff(fn, *, max_retries: int = 6, base_sleep: float = 2.0):
    """Retry transient HF rate-limit failures with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            if "429" not in msg and "Too Many Requests" not in msg:
                raise
            last_exc = exc
            sleep_s = base_sleep * (2 ** attempt)
            print(
                f"[retry] Hugging Face rate limit hit, sleeping {sleep_s:.1f}s...", flush=True)
            time.sleep(sleep_s)
    raise last_exc


def ensure_tokenized_sst2_dataset(config: ExperimentConfig) -> Path:
    """Ensure a tokenized SST-2 dataset exists on disk and return its path."""

    if config.tokenized_dataset_dir is None:
        raise ValueError("tokenized_dataset_dir must be set for SST-2 runs.")

    tokenized_dir = Path(config.tokenized_dataset_dir)
    train_dir = tokenized_dir / "train"
    val_dir = tokenized_dir / "validation"

    if train_dir.exists() and val_dir.exists():
        return tokenized_dir

    tokenized_dir.mkdir(parents=True, exist_ok=True)

    load_dataset, _, AutoTokenizer, _, _ = import_sst2_modules()

    raw_dataset = retry_with_backoff(
        lambda: load_dataset(
            "glue",
            "sst2",
            cache_dir=config.cache_dir,
            download_mode="reuse_dataset_if_exists",
            token=config.hf_token,
        )
    )

    tokenizer = retry_with_backoff(
        lambda: AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            token=config.hf_token,
        )
    )

    def tokenize_batch(batch: dict[str, list[object]]) -> dict[str, object]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=config.max_length,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        load_from_cache_file=True,
        desc="Tokenizing SST-2",
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])

    tokenized_dataset["train"].save_to_disk(str(train_dir))
    tokenized_dataset["validation"].save_to_disk(str(val_dir))
    return tokenized_dir


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Create SST-2 train and validation loaders from a saved tokenized dataset."""

    _, load_from_disk, AutoTokenizer, _, DataCollatorWithPadding = import_sst2_modules()

    tokenized_dir = ensure_tokenized_sst2_dataset(config)
    train_dataset = load_from_disk(str(tokenized_dir / "train"))
    val_dataset = load_from_disk(str(tokenized_dir / "validation"))

    tokenizer = retry_with_backoff(
        lambda: AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            token=config.hf_token,
        )
    )
    padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def collate_fn(features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([int(feature["label"])
                              for feature in features], dtype=torch.long)
        inputs = [{key: value for key, value in feature.items() if key != "label"}
                  for feature in features]
        batch = padder(inputs)
        batch["labels"] = labels
        return batch

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(config: ExperimentConfig, device: torch.device) -> nn.Module:
    """Build a DistilBERT classifier for SST-2."""

    _, _, _, AutoModelForSequenceClassification, _ = import_sst2_modules()
    model = retry_with_backoff(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            token=config.hf_token,
        )
    )
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
    method: str,
    random_seed: int,
    context_mode: str,
) -> object | None:
    """Build one LR or noise controller, or return None for fixed axes."""

    if len(arm_values) == 1:
        return None
    if method == "RandomScheduler":
        return RandomController(n_arms=len(arm_values), random_seed=random_seed)
    if method != "AdaptiveScheduler":
        raise ValueError(f"Unsupported structured method: {method}")
    bucket_names = build_context_bucket_names(context_mode)
    if bucket_names is not None:
        return BucketedContextualController(
            n_arms=len(arm_values),
            bucket_names=bucket_names,
            random_seed=random_seed,
            prior_from_global=True,
        )
    return DiscountedUCBController(n_arms=len(arm_values), random_seed=random_seed)


def build_method_components(
    config: ExperimentConfig,
    model: nn.Module,
    total_training_steps: int,
) -> TrainingComponents:
    """Build optimizer, controller, and episode manager objects for one run."""

    parameters = model.parameters()
    if config.method == "AdamW":
        return TrainingComponents(
            optimizer=torch.optim.AdamW(
                parameters, lr=config.lr, weight_decay=config.weight_decay),
            episode_manager=None,
            method_name="AdamW",
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
            [lr_candidates[0]], [noise_candidates[0]])[0].name,
        lr_multiplier=lr_candidates[0],
        noise_std=noise_candidates[0],
    )
    optimizer = AdaptiveModeAdamW(
        parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
        mode=initial_mode,
        noise_seed=config.seed,
    )
    lr_controller = build_axis_controller(
        arm_values=lr_candidates,
        method=config.method,
        random_seed=config.seed,
        context_mode=config.context_mode,
    )
    if config.structured_control_mode == "independent":
        noise_controller: object | list[object] = build_axis_controller(
            arm_values=noise_candidates,
            method=config.method,
            random_seed=config.seed + 101,
            context_mode=config.context_mode,
        )
    elif len(lr_candidates) == 1:
        noise_controller = build_axis_controller(
            arm_values=noise_candidates,
            method=config.method,
            random_seed=config.seed + 101,
            context_mode=config.context_mode,
        )
    else:
        noise_controller = [
            build_axis_controller(
                arm_values=noise_candidates,
                method=config.method,
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
        method_name=config.method,
        controller_logs=build_structured_controller_logs_container(
            method_name=config.method,
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
    total_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the configured step-level scheduler."""

    if config.lr_scheduler == "none":
        return None
    schedule_span = config.scheduler_t_max or total_training_steps
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_span)
    if config.lr_scheduler == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=build_step_linear_lr_lambda(schedule_span),
        )
    if config.lr_scheduler == "warmup_linear":
        warmup_steps = int(
            config.warmup_epochs *
            (total_training_steps / max(config.epochs, 1))
        )
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=build_step_warmup_linear_lr_lambda(
                total_steps=schedule_span,
                warmup_steps=warmup_steps,
            ),
        )
    raise ValueError(f"Unsupported lr scheduler: {config.lr_scheduler}")


def build_step_linear_lr_lambda(total_steps: int) -> Any:
    """Return a simple optimizer-step linear decay schedule."""

    def lr_lambda(step_index: int) -> float:
        progress = min(max(step_index / max(total_steps, 1), 0.0), 1.0)
        return max(0.0, 1.0 - progress)

    return lr_lambda


def build_step_warmup_linear_lr_lambda(total_steps: int, warmup_steps: int) -> Any:
    """Return a simple optimizer-step warmup-then-linear schedule."""

    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    def lr_lambda(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return min((step_index + 1) / warmup_steps, 1.0)
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_progress = min(
            max((step_index - warmup_steps) / decay_steps, 0.0), 1.0)
        return max(0.0, 1.0 - decay_progress)

    return lr_lambda


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    components: TrainingComponents,
    global_step: int,
    flop_accumulator: FlopAccumulator,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float, int]:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "labels"
        }
        if components.episode_manager is not None:
            mode = components.episode_manager.on_step_start(global_step)
            adaptive_optimizer = components.optimizer
            if not isinstance(adaptive_optimizer, AdaptiveModeAdamW):
                raise TypeError(
                    "Episode-managed methods require AdaptiveModeAdamW.")
            adaptive_optimizer.set_mode(mode)

        components.optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs, labels=labels)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                "NLP classification training step did not produce a loss.")
        loss.backward()
        components.optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        batch_size = int(labels.size(0))
        seq_len = int(model_inputs["input_ids"].shape[1])
        flop_accumulator.add_train_batch(
            estimate_distilbert_sst2_train_flops_per_batch(
                batch_size=batch_size, seq_len=seq_len)
        )

        predictions = outputs.logits.argmax(dim=1)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        total_correct += int((predictions == labels).sum().item())

        if components.episode_manager is not None:
            update_norm = (
                components.optimizer.compute_update_norm()
                if isinstance(components.optimizer, AdaptiveModeAdamW)
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
) -> float:
    """Evaluate validation accuracy."""

    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "labels"
        }
        outputs = model(**model_inputs)
        batch_size = int(labels.size(0))
        seq_len = int(model_inputs["input_ids"].shape[1])
        flop_accumulator.add_eval_batch(
            estimate_distilbert_sst2_eval_flops_per_batch(
                batch_size=batch_size, seq_len=seq_len)
        )
        predictions = outputs.logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += batch_size
    return correct / max(total, 1)


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
) -> RunResult:
    """Create the final JSON-serializable run summary."""

    episode_logs = None
    if components.episode_manager is not None:
        episode_logs = build_episode_logs(
            components.episode_manager.get_logs())

    config_dict = asdict(config)
    config_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["pin_memory"] = torch.cuda.is_available()
    config_dict["base_lr"] = config.lr
    config_dict["base_weight_decay"] = config.weight_decay
    config_dict["lr_scheduler"] = config.lr_scheduler
    config_dict["scheduler_t_max"] = config.scheduler_t_max
    config_dict["scheduler_active"] = config.lr_scheduler != "none"
    config_dict["scheduler_step_unit"] = "optimizer_step"
    config_dict["scheduler_span_steps"] = config.scheduler_t_max or total_steps
    config_dict["warmup_epochs"] = config.warmup_epochs
    label_noise_rate = float(getattr(config, "label_noise_rate", 0.0))
    config_dict["label_noise_rate"] = label_noise_rate
    config_dict["label_noise_seed"] = int(getattr(config, "label_noise_seed", 42))
    config_dict["label_noise_type"] = "symmetric" if label_noise_rate > 0.0 else "none"
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
        dataset_name=config.dataset_name,
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
    """Run one full SST-2 experiment."""

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(config)
    total_training_steps = len(train_loader) * config.epochs
    model = build_model(config, device)
    components = build_method_components(
        config,
        model,
        total_training_steps=total_training_steps,
    )
    lr_scheduler = build_lr_scheduler(
        config,
        components.optimizer,
        total_training_steps=total_training_steps,
    )

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
            components=components,
            global_step=global_step,
            flop_accumulator=flop_accumulator,
            lr_scheduler=lr_scheduler,
        )
        final_val_accuracy = evaluate_accuracy(
            model, val_loader, device, flop_accumulator)
        best_val_accuracy = max(best_val_accuracy, final_val_accuracy)
        train_losses.append(final_train_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(final_val_accuracy)
        epoch_wall_clock_seconds.append(time.perf_counter() - epoch_start_time)
        epoch_tflops.append(flop_accumulator.finish_epoch())
        print_epoch_status(
            method_name=config.method,
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
    )


def build_structured_controller_logs_container(
    *,
    method_name: str,
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
    if method_name == "RandomScheduler":
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

    if config.pretokenize_only:
        path = ensure_tokenized_sst2_dataset(config)
        print(f"Saved tokenized SST-2 dataset to {path}")
        return

    result = run_experiment(config)
    save_run_result(result, config.output)
    print(f"Saved result to {config.output}")


if __name__ == "__main__":
    main()
