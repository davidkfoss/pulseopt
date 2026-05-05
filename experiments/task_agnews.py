"""Run an AG News experiment with AdamW or AEES structured scheduling."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import random
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
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

from experiments.task_sst2 import (  # noqa: E402
    DEFAULT_CONFIG as SST2_DEFAULT_CONFIG,
    ExperimentConfig,
    build_lr_scheduler,
    build_method_components,
    build_result,
    collect_controller_snapshot_if_episode_closed,
    evaluate_accuracy,
    get_completed_episode_count,
    parse_lr_candidates,
    parse_noise_candidates,
    print_epoch_status,
    retry_with_backoff,
    set_seed,
    train_one_epoch,
)
from experiments.utils.flops import FlopAccumulator  # noqa: E402
from experiments.utils.metrics import RunResult  # noqa: E402
from experiments.utils.results import save_run_result  # noqa: E402

AGNEWS_VALIDATION_FRACTION = 0.1
AGNEWS_SPLIT_SEED = 42
AGNEWS_NUM_LABELS = 4

DEFAULT_CONFIG: dict[str, object] = {
    **SST2_DEFAULT_CONFIG,
    "task_name": "agnews",
    "dataset_name": "AG News",
    "tokenized_dataset_dir": str(PROJECT_ROOT / "data" / "agnews_tokenized"),
    "validation_fraction": AGNEWS_VALIDATION_FRACTION,
}


def parse_args() -> ExperimentConfig:
    """Parse the AG News CLI into the shared NLP experiment config."""

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
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(DEFAULT_CONFIG["weight_decay"]),
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
        "--label-noise-rate",
        type=float,
        default=0.0,
        help="Fraction of AG News training labels to corrupt.",
    )
    parser.add_argument(
        "--label-noise-seed",
        type=int,
        default=42,
        help="Seed used for deterministic AG News label corruption.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Defaults to results/agnews_<method>_seed<seed>.json",
    )
    parser.add_argument(
        "--tokenized-dataset-dir",
        type=str,
        default=str(DEFAULT_CONFIG["tokenized_dataset_dir"]),
    )
    parser.add_argument(
        "--pretokenize-only",
        action="store_true",
        help="Only build and save the tokenized AG News dataset, then exit.",
    )
    args = parser.parse_args()

    validate_args(args)
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
        label_noise_rate=args.label_noise_rate,
        label_noise_seed=args.label_noise_seed,
        tokenized_dataset_dir=args.tokenized_dataset_dir,
        pretokenize_only=bool(args.pretokenize_only),
        run_tag=args.run_tag,
        model_name=str(DEFAULT_CONFIG["model_name"]),
        task_name=str(DEFAULT_CONFIG["task_name"]),
        dataset_name=str(DEFAULT_CONFIG["dataset_name"]),
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate AG News CLI arguments."""

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
    if args.label_noise_rate < 0.0 or args.label_noise_rate >= 1.0:
        raise ValueError("--label-noise-rate must be in [0.0, 1.0).")


def build_default_output_path(method: str, seed: int) -> str:
    """Build the default JSON output path."""

    return str(PROJECT_ROOT / "results" / f"agnews_{method.lower()}_seed{seed}.json")


def format_noise_rate_tag(rate: float) -> str:
    """Return a filesystem-safe tag for one noise rate."""

    return f"{rate:g}".replace(".", "p")


def get_train_split_dir(tokenized_dir: Path, config: ExperimentConfig) -> Path:
    """Return the cache path for the clean or noisy AG News training split."""

    base_name = f"train_splitseed{AGNEWS_SPLIT_SEED}"
    if config.label_noise_rate > 0.0:
        rate_tag = format_noise_rate_tag(float(config.label_noise_rate))
        return tokenized_dir / f"{base_name}_noise{rate_tag}_noiseseed{config.label_noise_seed}"
    return tokenized_dir / base_name


def get_validation_split_dir(tokenized_dir: Path) -> Path:
    """Return the cache path for the clean AG News validation split."""

    return tokenized_dir / f"validation_splitseed{AGNEWS_SPLIT_SEED}"


def import_agnews_modules() -> tuple[object, object, object, object, object]:
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
            "datasets and transformers are required for AG News experiments."
        ) from exc
    return (
        load_dataset,
        load_from_disk,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
    )


def corrupt_agnews_labels(dataset, noise_rate: float, noise_seed: int):
    """Apply deterministic symmetric training-label noise to AG News."""

    if noise_rate <= 0.0:
        return dataset

    rng = random.Random(noise_seed)
    num_examples = len(dataset)
    num_corrupt = int(round(noise_rate * num_examples))
    if num_corrupt <= 0:
        return dataset

    corrupted_indices = set(rng.sample(range(num_examples), num_corrupt))
    replacement_labels: dict[int, int] = {}
    for index in sorted(corrupted_indices):
        original_label = int(dataset[index]["label"])
        choices = [label for label in range(AGNEWS_NUM_LABELS) if label != original_label]
        replacement_labels[index] = rng.choice(choices)

    def corrupt_example(example: dict[str, object], index: int) -> dict[str, object]:
        if index in replacement_labels:
            example["label"] = replacement_labels[index]
        return example

    noisy_dataset = dataset.map(
        corrupt_example,
        with_indices=True,
        desc=f"Applying {noise_rate:g} symmetric AG News label noise",
    )
    print(
        "Applied symmetric label noise to "
        f"{num_corrupt} / {num_examples} AG News training examples "
        f"(rate={noise_rate:g}, seed={noise_seed})."
    )
    return noisy_dataset


def ensure_tokenized_agnews_dataset(config: ExperimentConfig) -> Path:
    """Ensure a tokenized AG News dataset exists on disk and return its path."""

    if config.tokenized_dataset_dir is None:
        raise ValueError("tokenized_dataset_dir must be set for AG News runs.")

    tokenized_dir = Path(config.tokenized_dataset_dir)
    train_dir = get_train_split_dir(tokenized_dir, config)
    val_dir = get_validation_split_dir(tokenized_dir)
    test_dir = tokenized_dir / "test"
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        return tokenized_dir

    tokenized_dir.mkdir(parents=True, exist_ok=True)
    load_dataset, _, AutoTokenizer, _, _ = import_agnews_modules()
    raw_dataset = retry_with_backoff(
        lambda: load_dataset(
            "ag_news",
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
            batch["text"],
            truncation=True,
            max_length=config.max_length,
        )

    tokenized_dataset = raw_dataset.map(
        tokenize_batch,
        batched=True,
        load_from_cache_file=True,
        desc="Tokenizing AG News",
    )
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    train_val_split = tokenized_dataset["train"].train_test_split(
        test_size=AGNEWS_VALIDATION_FRACTION,
        seed=AGNEWS_SPLIT_SEED,
        shuffle=True,
    )
    clean_train_dataset = train_val_split["train"]
    train_dataset_to_save = corrupt_agnews_labels(
        clean_train_dataset,
        config.label_noise_rate,
        config.label_noise_seed,
    )
    if not train_dir.exists():
        reset_generated_split_dir(train_dir)
        train_dataset_to_save.save_to_disk(str(train_dir))
    if not val_dir.exists():
        reset_generated_split_dir(val_dir)
        train_val_split["test"].save_to_disk(str(val_dir))
    if not test_dir.exists():
        tokenized_dataset["test"].save_to_disk(str(test_dir))
    return tokenized_dir


def reset_generated_split_dir(path: Path) -> None:
    """Remove a partial generated split directory before saving it again."""

    if path.exists():
        shutil.rmtree(path)


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Create AG News train and validation loaders from a saved tokenized dataset."""

    _, load_from_disk, AutoTokenizer, _, DataCollatorWithPadding = import_agnews_modules()
    tokenized_dir = ensure_tokenized_agnews_dataset(config)
    train_dataset = load_from_disk(str(get_train_split_dir(tokenized_dir, config)))
    val_dataset = load_from_disk(str(get_validation_split_dir(tokenized_dir)))
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
        labels = torch.tensor(
            [int(feature["label"]) for feature in features],
            dtype=torch.long,
        )
        inputs = [
            {key: value for key, value in feature.items() if key != "label"}
            for feature in features
        ]
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


def build_model(config: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    """Build a DistilBERT classifier for AG News."""

    _, _, _, AutoModelForSequenceClassification, _ = import_agnews_modules()
    model = retry_with_backoff(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=4,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            token=config.hf_token,
            ignore_mismatched_sizes=True,
        )
    )
    return model.to(device)


def run_experiment(config: ExperimentConfig) -> RunResult:
    """Run one full AG News experiment."""

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
            model,
            val_loader,
            device,
            flop_accumulator,
        )
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
            components.episode_manager.get_logs()["episode_rewards"]
        )
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


def main() -> None:
    """Run one AG News experiment from the CLI and save the JSON result."""

    config = parse_args()
    if config.pretokenize_only:
        path = ensure_tokenized_agnews_dataset(config)
        print(f"Saved tokenized AG News dataset to {path}")
        return

    result = run_experiment(config)
    save_run_result(result, config.output)
    print(f"Saved result to {config.output}")


if __name__ == "__main__":
    main()
