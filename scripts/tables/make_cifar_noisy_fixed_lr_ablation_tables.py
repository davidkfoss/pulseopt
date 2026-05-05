#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

RESULTS_ROOT = Path("results/cifar_noisy")
OUTPUT_DIR = Path("results/tables/cifar_noisy")

TASK_ORDER = ["asym20", "sym20", "sym40"]
TASK_TITLES = {
    "asym20": "Asym. 20\\%",
    "sym20": "Sym. 20\\%",
    "sym40": "Sym. 40\\%",
}

OPTIMIZER_ORDER = ["AdamW", "SGD"]

# Folder names:
# cifar100_asym20_seed0
TASK_DIR_RE = re.compile(r"^cifar100_(asym20|sym20|sym40)_seed(\d+)$")

# File/run names inside each folder.
RUN_NAME_MAP = {
    # AdamW main methods
    "adamw_baseline": ("AdamW", "Baseline"),
    "adamw_aees_ep200_lr05102": ("AdamW", "AEES"),
    "adamw_cosine": ("AdamW", "Cosine"),
    "adamw_cosine_aees_ep200_lr05102": ("AdamW", "Cosine + AEES"),

    # AdamW fixed multipliers
    "adamw_fixed_lr05": ("AdamW", "Fixed 0.5"),
    "adamw_fixed_lr10": ("AdamW", "Fixed 1.0"),
    "adamw_fixed_lr20": ("AdamW", "Fixed 2.0"),

    # SGD main methods
    "sgd_baseline_lr01": ("SGD", "Baseline"),
    "sgd_aees_ep200_lr05102_base01": ("SGD", "AEES"),
    "sgd_cosine_lr01": ("SGD", "Cosine"),
    "sgd_cosine_aees_ep200_lr05102_base01": ("SGD", "Cosine + AEES"),

    # SGD fixed multipliers
    "sgd_fixed_lr05_base01": ("SGD", "Fixed 0.5"),
    "sgd_fixed_lr10_base01": ("SGD", "Fixed 1.0"),
    "sgd_fixed_lr20_base01": ("SGD", "Fixed 2.0"),
}

FIXED_METHODS = ["Fixed 0.5", "Fixed 1.0", "Fixed 2.0"]

DETAILED_METHOD_ORDER = [
    "Baseline",
    "Fixed 0.5",
    "Fixed 1.0",
    "Fixed 2.0",
    "AEES",
    "Cosine",
    "Cosine + AEES",
]


@dataclass
class RunMetrics:
    task: str
    seed: int
    optimizer: str
    method: str
    best_val: float
    final_val: float
    drop_val: float
    peak_epoch: float
    path: Path


@dataclass
class AggMetrics:
    n: int
    best_val_mean: float
    best_val_std: float
    final_val_mean: float
    final_val_std: float
    drop_val_mean: float
    drop_val_std: float
    peak_epoch_mean: float
    peak_epoch_std: float


def safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else math.nan


def safe_std(xs: List[float]) -> float:
    if not xs:
        return math.nan
    return stdev(xs) if len(xs) > 1 else 0.0


def pct(x: float) -> float:
    return 100.0 * x


def format_pm_pct(mean_val: float, std_val: float) -> str:
    if math.isnan(mean_val):
        return "--"
    if math.isnan(std_val):
        return f"{pct(mean_val):.1f}"
    return f"{pct(mean_val):.1f} $\\pm$ {pct(std_val):.1f}"


def format_pm_float(mean_val: float, std_val: float) -> str:
    if math.isnan(mean_val):
        return "--"
    if math.isnan(std_val):
        return f"{mean_val:.1f}"
    return f"{mean_val:.1f} $\\pm$ {std_val:.1f}"


def format_compact_triplet(a: AggMetrics) -> str:
    """Best / final / drop, all in percentage points."""
    return (
        f"{pct(a.best_val_mean):.1f} / "
        f"{pct(a.final_val_mean):.1f} / "
        f"{pct(a.drop_val_mean):.1f}"
    )


def is_likely_json_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if path.suffix in {".pt", ".pth", ".png", ".pdf", ".csv", ".tex", ".txt"}:
        return False

    try:
        with path.open("r") as f:
            return f.read(1) == "{"
    except Exception:
        return False


def get_val_accuracies(data: dict) -> List[float]:
    for key in ["val_accuracies", "val_accuracy", "validation_accuracies"]:
        vals = data.get(key)
        if isinstance(vals, list) and vals:
            return [float(v) for v in vals]
    return []


def get_best_val(data: dict, val_accuracies: List[float]) -> float:
    for key in ["best_val_accuracy", "best_validation_accuracy", "best_accuracy"]:
        if key in data:
            return float(data[key])
    return max(val_accuracies)


def get_final_val(data: dict, val_accuracies: List[float]) -> float:
    for key in ["final_val_accuracy", "final_validation_accuracy", "final_accuracy"]:
        if key in data:
            return float(data[key])
    return float(val_accuracies[-1])


def load_run(run_path: Path) -> Optional[RunMetrics]:
    task_dir = run_path.parent.name
    run_name = run_path.name

    task_match = TASK_DIR_RE.match(task_dir)
    if not task_match:
        return None

    task = task_match.group(1)
    seed = int(task_match.group(2))

    if run_name not in RUN_NAME_MAP:
        return None

    optimizer, method = RUN_NAME_MAP[run_name]

    try:
        data = json.loads(run_path.read_text())
    except Exception:
        return None

    val_accuracies = get_val_accuracies(data)
    if not val_accuracies:
        return None

    best_val = get_best_val(data, val_accuracies)
    final_val = get_final_val(data, val_accuracies)
    drop_val = best_val - final_val

    peak_epoch = float(
        max(range(len(val_accuracies)), key=lambda i: val_accuracies[i]) + 1
    )

    return RunMetrics(
        task=task,
        seed=seed,
        optimizer=optimizer,
        method=method,
        best_val=best_val,
        final_val=final_val,
        drop_val=drop_val,
        peak_epoch=peak_epoch,
        path=run_path,
    )


def collect_runs(root: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []

    for task_dir in root.glob("cifar100_*_seed*"):
        if not task_dir.is_dir():
            continue

        for child in task_dir.iterdir():
            if not is_likely_json_file(child):
                continue

            loaded = load_run(child)
            if loaded is not None:
                runs.append(loaded)

    return runs


def aggregate_runs(runs: List[RunMetrics]) -> Dict[Tuple[str, str, str], AggMetrics]:
    grouped: Dict[Tuple[str, str, str], List[RunMetrics]] = defaultdict(list)

    for run in runs:
        grouped[(run.task, run.optimizer, run.method)].append(run)

    out: Dict[Tuple[str, str, str], AggMetrics] = {}

    for key, group in grouped.items():
        best_vals = [r.best_val for r in group]
        final_vals = [r.final_val for r in group]
        drop_vals = [r.drop_val for r in group]
        peak_epochs = [r.peak_epoch for r in group]

        out[key] = AggMetrics(
            n=len(group),
            best_val_mean=safe_mean(best_vals),
            best_val_std=safe_std(best_vals),
            final_val_mean=safe_mean(final_vals),
            final_val_std=safe_std(final_vals),
            drop_val_mean=safe_mean(drop_vals),
            drop_val_std=safe_std(drop_vals),
            peak_epoch_mean=safe_mean(peak_epochs),
            peak_epoch_std=safe_std(peak_epochs),
        )

    return out


def print_seed_coverage(runs: List[RunMetrics]) -> None:
    grouped: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    for r in runs:
        grouped[(r.task, r.optimizer, r.method)].append(r.seed)

    print("\n===================================")
    print("NOISY CIFAR FIXED LR SEED COVERAGE")
    print("===================================")

    for task in TASK_ORDER:
        print(f"\n[{task}]")
        for optimizer in OPTIMIZER_ORDER:
            print(f"  {optimizer}")
            for method in DETAILED_METHOD_ORDER:
                seeds = sorted(set(grouped.get((task, optimizer, method), [])))
                if seeds:
                    print(f"    {method:<15} n={len(seeds)} seeds={seeds}")
                else:
                    print(f"    {method:<15} MISSING")


def best_fixed_for(
    agg: Dict[Tuple[str, str, str], AggMetrics],
    task: str,
    optimizer: str,
    metric_name: str = "best_val_mean",
) -> Optional[Tuple[str, AggMetrics]]:
    candidates: List[Tuple[str, AggMetrics]] = []

    for method in FIXED_METHODS:
        key = (task, optimizer, method)
        if key not in agg:
            continue
        candidates.append((method, agg[key]))

    if not candidates:
        return None

    return max(candidates, key=lambda x: getattr(x[1], metric_name))


def build_summary_table(agg: Dict[Tuple[str, str, str], AggMetrics]) -> str:
    lines: List[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{Noise} & \\multirow{2}{*}{Optimizer} "
        "& \\multirow{2}{*}{Best fixed} "
        "& \\multicolumn{4}{c}{Validation best / final / drop (pp)} \\\\"
    )
    lines.append(
        "& & & Best fixed & AEES & Cosine & Cosine + AEES \\\\"
    )
    lines.append("\\midrule")

    for task_idx, task in enumerate(TASK_ORDER):
        for opt_idx, optimizer in enumerate(OPTIMIZER_ORDER):
            task_col = TASK_TITLES[task] if opt_idx == 0 else ""

            best_fixed = best_fixed_for(agg, task, optimizer)

            if best_fixed is None:
                best_fixed_name = "--"
                best_fixed_cell = "--"
            else:
                best_fixed_name, best_fixed_agg = best_fixed
                best_fixed_name = best_fixed_name.replace(
                    "Fixed ", "$\\times$")
                best_fixed_cell = format_compact_triplet(best_fixed_agg)

            def get_cell(method: str) -> str:
                key = (task, optimizer, method)
                if key not in agg:
                    return "--"
                return format_compact_triplet(agg[key])

            lines.append(
                f"{task_col} & {optimizer} & {best_fixed_name} & "
                f"{best_fixed_cell} & "
                f"{get_cell('AEES')} & "
                f"{get_cell('Cosine')} & "
                f"{get_cell('Cosine + AEES')} \\\\"
            )

        if task_idx != len(TASK_ORDER) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Fixed learning-rate multiplier ablation summary for noisy \\cifar{}. "
        "For each noise setting and optimizer, the table reports the best fixed multiplier according to peak validation accuracy, "
        "using the same multiplier set available to the AEES controller. Validation entries are reported as best / final / drop, "
        "where drop is the best-to-final validation decrease in percentage points. This table tests whether the AEES peak gains under label noise exceed the best static LR rescaling baseline.}"
    )
    lines.append("\\label{tab:cifar_noisy_fixed_lr_ablation_summary}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_detailed_table(agg: Dict[Tuple[str, str, str], AggMetrics]) -> str:
    lines: List[str] = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lllccccc}")
    lines.append("\\toprule")
    lines.append(
        "Noise & Optimizer & Method & $n$ & Best (\\%) & Final (\\%) & Drop (pp) & Peak epoch \\\\"
    )
    lines.append("\\midrule")

    for task_idx, task in enumerate(TASK_ORDER):
        if task_idx > 0:
            lines.append("\\midrule")

        for optimizer in OPTIMIZER_ORDER:
            valid_methods = [
                m for m in DETAILED_METHOD_ORDER
                if (task, optimizer, m) in agg
            ]

            for method_idx, method in enumerate(valid_methods):
                a = agg[(task, optimizer, method)]

                noise_col = TASK_TITLES[task] if optimizer == OPTIMIZER_ORDER[0] and method_idx == 0 else ""
                opt_col = optimizer if method_idx == 0 else ""

                lines.append(
                    f"{noise_col} & {opt_col} & {method} & {a.n} & "
                    f"{format_pm_pct(a.best_val_mean, a.best_val_std)} & "
                    f"{format_pm_pct(a.final_val_mean, a.final_val_std)} & "
                    f"{format_pm_pct(a.drop_val_mean, a.drop_val_std)} & "
                    f"{format_pm_float(a.peak_epoch_mean, a.peak_epoch_std)} \\\\"
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Detailed fixed learning-rate multiplier ablation results for noisy \\cifar{}. "
        "Fixed multipliers use the same LR multiplier values available to the AEES controller. "
        "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
        "Drop denotes the best-to-final validation decrease in percentage points, where lower is better.}"
    )
    lines.append("\\label{tab:cifar_noisy_fixed_lr_ablation_detailed}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def write_summary_csv(
    agg: Dict[Tuple[str, str, str], AggMetrics],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "optimizer",
            "best_fixed_method",
            "best_fixed_best",
            "best_fixed_final",
            "best_fixed_drop",
            "aees_best",
            "aees_final",
            "aees_drop",
            "cosine_best",
            "cosine_final",
            "cosine_drop",
            "cosine_aees_best",
            "cosine_aees_final",
            "cosine_aees_drop",
        ])

        for task in TASK_ORDER:
            for optimizer in OPTIMIZER_ORDER:
                best_fixed = best_fixed_for(agg, task, optimizer)

                if best_fixed is None:
                    best_fixed_method = ""
                    best_fixed_a = None
                else:
                    best_fixed_method, best_fixed_a = best_fixed

                def values(method: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
                    key = (task, optimizer, method)
                    if key not in agg:
                        return None, None, None
                    a = agg[key]
                    return a.best_val_mean, a.final_val_mean, a.drop_val_mean

                writer.writerow([
                    task,
                    optimizer,
                    best_fixed_method,
                    None if best_fixed_a is None else best_fixed_a.best_val_mean,
                    None if best_fixed_a is None else best_fixed_a.final_val_mean,
                    None if best_fixed_a is None else best_fixed_a.drop_val_mean,
                    *values("AEES"),
                    *values("Cosine"),
                    *values("Cosine + AEES"),
                ])


def write_detailed_csv(
    agg: Dict[Tuple[str, str, str], AggMetrics],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "optimizer",
            "method",
            "n",
            "best_val_mean",
            "best_val_std",
            "final_val_mean",
            "final_val_std",
            "drop_val_mean",
            "drop_val_std",
            "peak_epoch_mean",
            "peak_epoch_std",
        ])

        for task in TASK_ORDER:
            for optimizer in OPTIMIZER_ORDER:
                for method in DETAILED_METHOD_ORDER:
                    key = (task, optimizer, method)
                    if key not in agg:
                        continue

                    a = agg[key]
                    writer.writerow([
                        task,
                        optimizer,
                        method,
                        a.n,
                        a.best_val_mean,
                        a.best_val_std,
                        a.final_val_mean,
                        a.final_val_std,
                        a.drop_val_mean,
                        a.drop_val_std,
                        a.peak_epoch_mean,
                        a.peak_epoch_std,
                    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(RESULTS_ROOT)
    if not runs:
        raise SystemExit(
            f"No matching noisy CIFAR JSON files found in {RESULTS_ROOT}")

    print_seed_coverage(runs)

    agg = aggregate_runs(runs)

    summary_table = build_summary_table(agg)
    summary_path = OUTPUT_DIR / "cifar_noisy_fixed_lr_ablation_summary_table.tex"
    summary_path.write_text(summary_table)
    print(f"Wrote {summary_path}")

    detailed_table = build_detailed_table(agg)
    detailed_path = OUTPUT_DIR / "cifar_noisy_fixed_lr_ablation_detailed_table.tex"
    detailed_path.write_text(detailed_table)
    print(f"Wrote {detailed_path}")

    summary_csv_path = OUTPUT_DIR / "cifar_noisy_fixed_lr_ablation_summary.csv"
    write_summary_csv(agg, summary_csv_path)
    print(f"Wrote {summary_csv_path}")

    detailed_csv_path = OUTPUT_DIR / "cifar_noisy_fixed_lr_ablation_detailed.csv"
    write_detailed_csv(agg, detailed_csv_path)
    print(f"Wrote {detailed_csv_path}")


if __name__ == "__main__":
    main()
