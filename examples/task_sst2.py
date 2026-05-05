"""SST-2 DistilBERT example using the pulseopt PyPI package.

Install once:
    pip install pulseopt torch transformers datasets

Run:
    python task_sst2.py --epochs 3 --output run.log
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pulseopt import AEES


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    model_name: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    raw = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize, batched=True, desc="Tokenizing SST-2")
    tokenized = tokenized.remove_columns(["sentence", "idx"])

    padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor(
            [int(f["label"]) for f in features], dtype=torch.long
        )
        inputs = [{k: v for k, v in f.items() if k != "label"} for f in features]
        batch = padder(inputs)
        batch["labels"] = labels
        return batch

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    return train_loader, val_loader


def build_model(model_name: str, device: torch.device) -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        preds = model(**batch).logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    model.train()
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    aees: AEES,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["labels"]

        aees.step_start(global_step)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        aees.step_end(loss)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((outputs.logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
        global_step += 1

    return running_loss / max(total, 1), correct / max(total, 1), global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument(
        "--lr-candidates", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--noise-candidates", type=float, nargs="+", default=[0.0, 0.005]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("./sst2_aees.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        args.model_name, args.batch_size, args.max_length, args.num_workers
    )
    model = build_model(args.model_name, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # AEES wraps the AdamW optimizer; it picks an LR multiplier and gradient-
    # noise std per episode using a per-axis bandit driven by log-loss reward.
    aees = AEES(
        optimizer,
        lr_candidates=args.lr_candidates,
        noise_candidates=args.noise_candidates,
        episode_length=args.episode_length,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    header = (
        f"SST-2 {args.model_name} + AEES (AdamW)\n"
        f"device={device} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}\n"
        f"lr_candidates={list(args.lr_candidates)} "
        f"noise_candidates={list(args.noise_candidates)}\n"
        f"episode_length={args.episode_length} seed={args.seed}\n"
        + "-" * 80
    )
    print(header, flush=True)
    log_lines.append(header)

    global_step = 0
    final_val_acc = 0.0
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, aees, optimizer, device, global_step
        )
        final_val_acc = evaluate(model, val_loader, device)
        line = (
            f"epoch={epoch + 1:>3d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={final_val_acc:.4f} "
            f"time_s={time.perf_counter() - epoch_start:.1f}"
        )
        print(line, flush=True)
        log_lines.append(line)

    aees.finalize()
    logs = aees.get_logs()
    summary = (
        "-" * 80 + "\n"
        f"total_time_s={time.perf_counter() - start_time:.1f} "
        f"episodes={len(logs.get('episode_rewards', []))} "
        f"final_val_acc={final_val_acc:.4f}"
    )
    print(summary, flush=True)
    log_lines.append(summary)

    args.output.write_text("\n".join(log_lines) + "\n")
    print(f"log written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
