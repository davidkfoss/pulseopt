#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("results/cifar_clean")
OUTPUT_DIR = Path("results/figures/cifar_clean")

LAST_N_EPOCHS = 20

FILE_RE = re.compile(r"^cifar_clean_(.+)_seed(\d+)\.json$")

RUN_NAME_MAP = {
    "adamw_none": "Fixed 1.0",
    "fixed_lr1_n0_none": "Fixed 1.0",
    "fixed_lr0p5_n0_none": "Fixed 0.5",
    "fixed_lr2_n0_none": "Fixed 2.0",
    "aees_none": "AEES",
}

METHOD_ORDER = [
    "Fixed 0.5",
    "Fixed 1.0",
    "Fixed 2.0",
    "AEES",
]


def get_val_accuracies(data: dict) -> List[float]:
    for key in ["val_accuracies", "val_accuracy", "validation_accuracies"]:
        vals = data.get(key)
        if isinstance(vals, list) and vals:
            return [float(v) for v in vals]
    return []


def load_runs() -> Dict[str, List[Tuple[int, List[float]]]]:
    runs: Dict[str, List[Tuple[int, List[float]]]] = defaultdict(list)

    for path in RESULTS_ROOT.iterdir():
        if not path.is_file() or path.suffix != ".json":
            continue

        match = FILE_RE.match(path.name)
        if not match:
            continue

        run_name = match.group(1)
        seed = int(match.group(2))

        if run_name not in RUN_NAME_MAP:
            continue

        method = RUN_NAME_MAP[run_name]

        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        vals = get_val_accuracies(data)
        if not vals:
            continue

        runs[method].append((seed, vals))

    for method in runs:
        runs[method] = sorted(runs[method], key=lambda x: x[0])

    return runs


def truncate_last_epochs(vals: List[float], n: int) -> List[float]:
    return vals[-n:] if len(vals) >= n else vals


def get_epoch_axis(vals: List[float], n: int) -> np.ndarray:
    start_epoch = len(vals) - len(truncate_last_epochs(vals, n)) + 1
    return np.arange(start_epoch, len(vals) + 1)


def plot_aees_individual_seeds(runs: Dict[str, List[Tuple[int, List[float]]]]) -> None:
    if "AEES" not in runs:
        raise RuntimeError("No AEES runs found.")

    plt.figure(figsize=(7.0, 4.2))

    for seed, vals in runs["AEES"]:
        y = np.array(truncate_last_epochs(vals, LAST_N_EPOCHS)) * 100.0
        x = get_epoch_axis(vals, LAST_N_EPOCHS)
        plt.plot(x, y, linewidth=1.2, alpha=0.8, label=f"Seed {seed}")

    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title(f"Clean CIFAR-100 AEES late-stage validation trajectories")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()

    out_png = OUTPUT_DIR / "cifar_clean_aees_late_stage_individual_seeds.png"
    out_pdf = OUTPUT_DIR / "cifar_clean_aees_late_stage_individual_seeds.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def align_last_n(
    method_runs: List[Tuple[int, List[float]]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not method_runs:
        raise RuntimeError("No runs provided.")

    min_len = min(len(vals) for _, vals in method_runs)
    use_n = min(n, min_len)

    arr = np.array([truncate_last_epochs(vals, use_n)
                   for _, vals in method_runs], dtype=float)
    arr *= 100.0

    # Use relative late-stage epoch index to avoid problems if some runs have different lengths.
    x = np.arange(-use_n + 1, 1)

    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(use_n)

    return x, mean, std


def plot_fixed_vs_aees_mean_std(runs: Dict[str, List[Tuple[int, List[float]]]]) -> None:
    plt.figure(figsize=(7.0, 4.2))

    plotted_any = False

    for method in METHOD_ORDER:
        if method not in runs:
            print(f"Skipping missing method: {method}")
            continue

        x, y_mean, y_std = align_last_n(runs[method], LAST_N_EPOCHS)

        plt.plot(x, y_mean, linewidth=1.8, label=method)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15)

        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No methods found for fixed-vs-AEES plot.")

    plt.xlabel("Epochs from final epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Clean CIFAR-100 fixed multipliers vs AEES")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_png = OUTPUT_DIR / "cifar_clean_fixed_vs_aees_late_stage_mean_std.png"
    out_pdf = OUTPUT_DIR / "cifar_clean_fixed_vs_aees_late_stage_mean_std.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def print_coverage(runs: Dict[str, List[Tuple[int, List[float]]]]) -> None:
    print("\n==============================")
    print("CLEAN CIFAR TRAJECTORY COVERAGE")
    print("==============================")

    for method in METHOD_ORDER:
        method_runs = runs.get(method, [])
        seeds = [seed for seed, _ in method_runs]
        lengths = [len(vals) for _, vals in method_runs]

        if not method_runs:
            print(f"{method:<12} MISSING")
        else:
            print(
                f"{method:<12} n={len(method_runs)} "
                f"seeds={seeds} "
                f"epochs={sorted(set(lengths))}"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    print_coverage(runs)

    plot_aees_individual_seeds(runs)
    plot_fixed_vs_aees_mean_std(runs)


if __name__ == "__main__":
    main()
