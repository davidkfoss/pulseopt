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
    "asym20": "CIFAR-100 Asymmetric 20\\% Noise",
    "sym20": "CIFAR-100 Symmetric 20\\% Noise",
    "sym40": "CIFAR-100 Symmetric 40\\% Noise",
}
TASK_SHORT = {
    "asym20": "Asym. 20\\%",
    "sym20": "Sym. 20\\%",
    "sym40": "Sym. 40\\%",
}

OPTIMIZER_ORDER = ["AdamW", "SGD"]
METHOD_ORDER = ["Baseline", "AEES", "Cosine", "Cosine + AEES"]

RUN_NAME_MAP = {
    "adamw_baseline": ("AdamW", "Baseline"),
    "adamw_aees_ep200_lr05102": ("AdamW", "AEES"),
    "adamw_cosine": ("AdamW", "Cosine"),
    "adamw_cosine_aees_ep200_lr05102": ("AdamW", "Cosine + AEES"),
    "sgd_baseline_lr01": ("SGD", "Baseline"),
    "sgd_aees_ep200_lr05102_base01": ("SGD", "AEES"),
    "sgd_cosine_lr01": ("SGD", "Cosine"),
    "sgd_cosine_aees_ep200_lr05102_base01": ("SGD", "Cosine + AEES"),
}

TASK_DIR_RE = re.compile(r"^cifar100_(asym20|sym20|sym40)_seed(\d+)$")


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
    corr_noisy: float
    corr_clean: float
    train_clean: Optional[float]
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
    corr_noisy_mean: float
    corr_noisy_std: float
    corr_clean_mean: float
    corr_clean_std: float
    train_clean_mean: Optional[float]
    train_clean_std: Optional[float]


def safe_mean(xs: List[float]) -> Optional[float]:
    return mean(xs) if xs else None


def safe_std(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
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


def format_optional_pm_pct(mean_val: Optional[float], std_val: Optional[float]) -> str:
    if mean_val is None or math.isnan(mean_val):
        return "--"
    if std_val is None or math.isnan(std_val):
        return f"{pct(mean_val):.1f}"
    return f"{pct(mean_val):.1f} $\\pm$ {pct(std_val):.1f}"


def is_likely_json_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if path.suffix in {".pt", ".pth", ".png", ".pdf", ".csv", ".tex", ".txt"}:
        return False
    try:
        with path.open("r") as f:
            first = f.read(1)
        return first == "{"
    except Exception:
        return False


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

    diagnostics = data.get("config", {}).get("diagnostics", {})
    subset = diagnostics.get("final_subset_accuracies", {})
    if not subset:
        return None

    val_accuracies = data.get("val_accuracies", [])
    if not val_accuracies:
        return None

    best_val = float(data["best_val_accuracy"])
    final_val = float(data["final_val_accuracy"])
    drop_val = best_val - final_val

    peak_epoch = float(max(range(len(val_accuracies)),
                       key=lambda i: val_accuracies[i]) + 1)

    return RunMetrics(
        task=task,
        seed=seed,
        optimizer=optimizer,
        method=method,
        best_val=best_val,
        final_val=final_val,
        drop_val=drop_val,
        peak_epoch=peak_epoch,
        corr_noisy=subset["train_corrupted_accuracy_vs_noisy_labels"],
        corr_clean=subset["train_corrupted_accuracy_vs_clean_labels"],
        train_clean=subset.get("train_clean_accuracy"),
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
        corr_noisy = [r.corr_noisy for r in group]
        corr_clean = [r.corr_clean for r in group]
        train_clean = [
            r.train_clean for r in group if r.train_clean is not None]

        out[key] = AggMetrics(
            n=len(group),
            best_val_mean=safe_mean(best_vals) or math.nan,
            best_val_std=safe_std(best_vals) or math.nan,
            final_val_mean=safe_mean(final_vals) or math.nan,
            final_val_std=safe_std(final_vals) or math.nan,
            drop_val_mean=safe_mean(drop_vals) or math.nan,
            drop_val_std=safe_std(drop_vals) or math.nan,
            peak_epoch_mean=safe_mean(peak_epochs) or math.nan,
            peak_epoch_std=safe_std(peak_epochs) or math.nan,
            corr_noisy_mean=safe_mean(corr_noisy) or math.nan,
            corr_noisy_std=safe_std(corr_noisy) or math.nan,
            corr_clean_mean=safe_mean(corr_clean) or math.nan,
            corr_clean_std=safe_std(corr_clean) or math.nan,
            train_clean_mean=safe_mean(train_clean),
            train_clean_std=safe_std(train_clean),
        )

    return out


def print_seed_coverage(runs: List[RunMetrics]) -> None:
    grouped: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for r in runs:
        grouped[(r.task, r.optimizer, r.method)].append(r.seed)

    print("\n==============================")
    print("NOISY CIFAR SEED COVERAGE")
    print("==============================")

    missing_any = False

    for task in TASK_ORDER:
        print(f"\n[{task}]")
        for optimizer in OPTIMIZER_ORDER:
            print(f"  {optimizer}")
            for method in METHOD_ORDER:
                seeds = sorted(set(grouped.get((task, optimizer, method), [])))
                if seeds:
                    print(f"    {method:<14} n={len(seeds)} seeds={seeds}")
                else:
                    missing_any = True
                    print(f"    {method:<14} MISSING")

    if missing_any:
        print("\nWARNING: Some expected task/optimizer/method combinations are missing.")
    else:
        print("\nAll expected task/optimizer/method combinations were found.")


def rank_methods_for_metric(
    agg: Dict[Tuple[str, str, str], AggMetrics],
    task: str,
    optimizer: str,
    metric_name: str,
    higher_is_better: bool = True,
) -> Dict[str, str]:
    vals = []
    for method in METHOD_ORDER:
        key = (task, optimizer, method)
        if key not in agg:
            continue
        vals.append((method, getattr(agg[key], metric_name)))

    vals.sort(key=lambda x: x[1], reverse=higher_is_better)

    styles = {m: "" for m in METHOD_ORDER}
    if len(vals) >= 1:
        styles[vals[0][0]] = "best"
    if len(vals) >= 2:
        styles[vals[1][0]] = "second"

    return styles


def styled_cell(text: str, style: str) -> str:
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\emph{{{text}}}"
    return text


def caption_for_task(task: str) -> str:
    if task == "asym20":
        return (
            "Noisy-label CIFAR-100 results under 20\\% asymmetric label noise. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    if task == "sym20":
        return (
            "Noisy-label CIFAR-100 results under 20\\% symmetric label noise. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    if task == "sym40":
        return (
            "Noisy-label CIFAR-100 results under 40\\% symmetric label noise. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    raise ValueError(f"Unknown task: {task}")


def build_task_table(
    task: str,
    agg: Dict[Tuple[str, str, str], AggMetrics],
) -> str:
    lines: List[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{Optimizer} & \\multirow{2}{*}{Method} "
        "& \\multicolumn{4}{c}{Validation} \\\\"
    )
    lines.append("& & Best (\\%) & Final (\\%) & Drop (pp) & Peak epoch \\\\")
    lines.append("\\midrule")

    for optimizer_idx, optimizer in enumerate(OPTIMIZER_ORDER):
        best_styles = rank_methods_for_metric(
            agg, task, optimizer, "best_val_mean", higher_is_better=True
        )
        final_styles = rank_methods_for_metric(
            agg, task, optimizer, "final_val_mean", higher_is_better=True
        )
        drop_styles = rank_methods_for_metric(
            agg, task, optimizer, "drop_val_mean", higher_is_better=False
        )

        valid_methods = [m for m in METHOD_ORDER if (
            task, optimizer, m) in agg]
        n_rows = len(valid_methods)
        first_row = True

        for method in valid_methods:
            a = agg[(task, optimizer, method)]

            best_cell = styled_cell(
                format_pm_pct(a.best_val_mean, a.best_val_std),
                best_styles[method],
            )
            final_cell = styled_cell(
                format_pm_pct(a.final_val_mean, a.final_val_std),
                final_styles[method],
            )
            drop_cell = styled_cell(
                format_pm_pct(a.drop_val_mean, a.drop_val_std),
                drop_styles[method],
            )
            peak_epoch_cell = format_pm_float(
                a.peak_epoch_mean, a.peak_epoch_std)

            opt_col = f"\\multirow{{{n_rows}}}{{*}}{{{optimizer}}}" if first_row else ""

            lines.append(
                f"{opt_col} & {method} & {best_cell} & {final_cell} & {drop_cell} & {peak_epoch_cell} \\\\"
            )
            first_row = False

        if optimizer_idx != len(OPTIMIZER_ORDER) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        f"\\caption{{{caption_for_task(task)} "
        "Within each optimizer block, bold and italics mark the best and second-best values, respectively.}}"
    )
    lines.append(f"\\label{{tab:cifar_{task}_main}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_detailed_appendix_table(
    agg: Dict[Tuple[str, str, str], AggMetrics],
) -> str:
    lines: List[str] = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lllcccccccc}")
    lines.append("\\toprule")
    lines.append(
        "Noise & Optimizer & Method & $n$ & Best & Final & Drop & Peak epoch "
        "& Corr. noisy & Corr. clean & Train clean \\\\"
    )
    lines.append(
        "& & & & (\\%) & (\\%) & (pp) & "
        "& (\\%) & (\\%) & (\\%) \\\\"
    )
    lines.append("\\midrule")

    first_task = True

    for task in TASK_ORDER:
        if not first_task:
            lines.append("\\midrule")
        first_task = False

        for optimizer in OPTIMIZER_ORDER:
            valid_methods = [m for m in METHOD_ORDER if (
                task, optimizer, m) in agg]
            for method_idx, method in enumerate(valid_methods):
                a = agg[(task, optimizer, method)]

                noise_col = TASK_SHORT[task] if optimizer == OPTIMIZER_ORDER[0] and method_idx == 0 else ""
                optimizer_col = optimizer if method_idx == 0 else ""

                lines.append(
                    f"{noise_col} & "
                    f"{optimizer_col} & "
                    f"{method} & "
                    f"{a.n} & "
                    f"{format_pm_pct(a.best_val_mean, a.best_val_std)} & "
                    f"{format_pm_pct(a.final_val_mean, a.final_val_std)} & "
                    f"{format_pm_pct(a.drop_val_mean, a.drop_val_std)} & "
                    f"{format_pm_float(a.peak_epoch_mean, a.peak_epoch_std)} & "
                    f"{format_pm_pct(a.corr_noisy_mean, a.corr_noisy_std)} & "
                    f"{format_pm_pct(a.corr_clean_mean, a.corr_clean_std)} & "
                    f"{format_optional_pm_pct(a.train_clean_mean, a.train_clean_std)} \\\\"
                )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Detailed noisy-label CIFAR-100 results. "
        "The table reports the number of seeds $n$, validation metrics, best-to-final validation drop, peak epoch, "
        "and diagnostic training-subset accuracies. Corr. noisy is accuracy on corrupted training examples measured against their corrupted labels; "
        "Corr. clean is accuracy on the same examples measured against their original clean labels. "
        "Train clean is accuracy on the uncorrupted training subset. These diagnostics are supplementary and are used to support the interpretation of memorization and robustness.}"
    )
    lines.append("\\label{tab:cifar_noisy_detailed_appendix}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def write_csv(
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
            "corr_noisy_mean",
            "corr_noisy_std",
            "corr_clean_mean",
            "corr_clean_std",
            "train_clean_mean",
            "train_clean_std",
        ])

        for task in TASK_ORDER:
            for optimizer in OPTIMIZER_ORDER:
                for method in METHOD_ORDER:
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
                        a.corr_noisy_mean,
                        a.corr_noisy_std,
                        a.corr_clean_mean,
                        a.corr_clean_std,
                        a.train_clean_mean,
                        a.train_clean_std,
                    ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(RESULTS_ROOT)
    if not runs:
        raise SystemExit("No matching noisy CIFAR JSON result files found.")

    print_seed_coverage(runs)

    agg = aggregate_runs(runs)

    for task in TASK_ORDER:
        table_tex = build_task_table(task, agg)
        out_path = OUTPUT_DIR / f"cifar_{task}_main_table.tex"
        out_path.write_text(table_tex)
        print(f"Wrote {out_path}")

    combined = "\n\n".join(build_task_table(task, agg) for task in TASK_ORDER)
    combined_path = OUTPUT_DIR / "cifar_noisy_main_tables_all.tex"
    combined_path.write_text(combined)
    print(f"Wrote {combined_path}")

    detailed = build_detailed_appendix_table(agg)
    detailed_path = OUTPUT_DIR / "cifar_noisy_detailed_appendix_table.tex"
    detailed_path.write_text(detailed)
    print(f"Wrote {detailed_path}")

    all_tables = combined + "\n\n" + detailed
    all_tables_path = OUTPUT_DIR / "cifar_noisy_tables_all.tex"
    all_tables_path.write_text(all_tables)
    print(f"Wrote {all_tables_path}")

    csv_path = OUTPUT_DIR / "cifar_noisy_main_tables_summary.csv"
    write_csv(agg, csv_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
