"""CIFAR-100 ResNet-18 example using the pulseopt PyPI package.

Install once:
    pip install pulseopt torch torchvision

Run:
    python task_cifar100.py --epochs 10 --output run.log
    python task_cifar100.py --optimizer SGD --lr 0.1 --output run_sgd.log
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
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    train_tf = transforms.Compose(
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
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    train_set = datasets.CIFAR100(root=str(data_dir), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=test_tf)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    """ResNet-18 with CIFAR-style first conv (3x3 stride 1) and no maxpool."""
    from torchvision import models

    model = models.resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def build_optimizer(
    name: str,
    parameters,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    if name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "SGD":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(inputs).argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.size(0))
    model.train()
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    aees: AEES,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        aees.step_start(global_step)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        aees.step_end(loss)

        running_loss += float(loss.item()) * targets.size(0)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.size(0))
        global_step += 1

    return running_loss / max(total, 1), correct / max(total, 1), global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["AdamW", "SGD"], default="AdamW")
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--lr-candidates", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--noise-candidates", type=float, nargs="+", default=[0.0, 0.005])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("./cifar100_aees.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(device)
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        args.lr,
        args.weight_decay,
        args.momentum,
    )
    criterion = nn.CrossEntropyLoss()

    # Wrap any torch optimizer with AEES — it adds a per-axis bandit that picks
    # an LR multiplier and gradient-noise std per episode based on log-loss
    # improvement. step_start/step_end bracket each training step.
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
        f"CIFAR-100 ResNet-18 + AEES ({args.optimizer})\n"
        f"device={device} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}\n"
        f"lr_candidates={list(args.lr_candidates)} "
        f"noise_candidates={list(args.noise_candidates)}\n"
        f"episode_length={args.episode_length} seed={args.seed}\n" + "-" * 80
    )
    print(header, flush=True)
    log_lines.append(header)

    global_step = 0
    final_val_acc = 0.0
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, aees, optimizer, criterion, device, global_step
        )
        final_val_acc = evaluate(model, test_loader, device)
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
