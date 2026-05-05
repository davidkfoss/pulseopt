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
from typing import Dict, List, Optional

RESULTS_ROOT = Path("results/cifar_clean")
OUTPUT_DIR = Path("results/tables/cifar_clean")

FILE_RE = re.compile(r"^cifar_clean_(.+)_seed(\d+)\.json$")

MAIN_METHOD_ORDER = [
    "AdamW",
    "AdamW + Cosine",
    "AdamW + Linear",
    "AEES",
    "AEES + Cosine",
    "AEES + Linear",
]

ABLATION_METHOD_ORDER = [
    "Fixed 1.0",
    "Fixed 0.5",
    "Fixed 2.0",
    "AEES",
]

RUN_NAME_MAP = {
    # Main clean CIFAR methods
    "adamw_none": "AdamW",
    "adamw_cosine": "AdamW + Cosine",
    "adamw_linear": "AdamW + Linear",
    "aees_none": "AEES",
    "aees_cosine": "AEES + Cosine",
    "aees_linear": "AEES + Linear",

    # Fixed LR multiplier ablations
    "fixed_lr1_n0_none": "Fixed 1.0",
    "fixed_lr0p5_n0_none": "Fixed 0.5",
    "fixed_lr2_n0_none": "Fixed 2.0",
}


@dataclass
class RunMetrics:
    seed: int
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


def styled_cell(text: str, style: str) -> str:
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\emph{{{text}}}"
    return text


def is_likely_json_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    if path.suffix != ".json":
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


def load_run(path: Path) -> Optional[RunMetrics]:
    match = FILE_RE.match(path.name)
    if not match:
        return None

    run_name = match.group(1)
    seed = int(match.group(2))

    if run_name not in RUN_NAME_MAP:
        return None

    method = RUN_NAME_MAP[run_name]

    try:
        data = json.loads(path.read_text())
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
        seed=seed,
        method=method,
        best_val=best_val,
        final_val=final_val,
        drop_val=drop_val,
        peak_epoch=peak_epoch,
        path=path,
    )


def collect_runs(root: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []

    for path in root.iterdir():
        if not is_likely_json_file(path):
            continue

        run = load_run(path)
        if run is not None:
            runs.append(run)

    # If no explicit fixed-lr1 files exist, use AdamW baseline as Fixed 1.0.
    # This is correct when AdamW baseline uses the same base LR and no scheduler.
    baseline_runs = [r for r in runs if r.method == "AdamW"]
    fixed_lr1_exists = any(r.method == "Fixed 1.0" for r in runs)

    if baseline_runs and not fixed_lr1_exists:
        for r in baseline_runs:
            runs.append(
                RunMetrics(
                    seed=r.seed,
                    method="Fixed 1.0",
                    best_val=r.best_val,
                    final_val=r.final_val,
                    drop_val=r.drop_val,
                    peak_epoch=r.peak_epoch,
                    path=r.path,
                )
            )

    return runs


def aggregate_runs(runs: List[RunMetrics]) -> Dict[str, AggMetrics]:
    grouped: Dict[str, List[RunMetrics]] = defaultdict(list)

    for run in runs:
        grouped[run.method].append(run)

    out: Dict[str, AggMetrics] = {}

    for method, group in grouped.items():
        best_vals = [r.best_val for r in group]
        final_vals = [r.final_val for r in group]
        drop_vals = [r.drop_val for r in group]
        peak_epochs = [r.peak_epoch for r in group]

        out[method] = AggMetrics(
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
    grouped: Dict[str, List[int]] = defaultdict(list)

    for r in runs:
        grouped[r.method].append(r.seed)

    print("\n==============================")
    print("CLEAN CIFAR SEED COVERAGE")
    print("==============================")

    all_methods = sorted(set(MAIN_METHOD_ORDER + ABLATION_METHOD_ORDER))

    for method in all_methods:
        seeds = sorted(set(grouped.get(method, [])))
        if seeds:
            print(f"{method:<18} n={len(seeds)} seeds={seeds}")
        else:
            print(f"{method:<18} MISSING")


def rank_methods_for_metric(
    agg: Dict[str, AggMetrics],
    method_order: List[str],
    metric_name: str,
    higher_is_better: bool,
) -> Dict[str, str]:
    vals = []

    for method in method_order:
        if method not in agg:
            continue

        val = getattr(agg[method], metric_name)
        if math.isnan(val):
            continue

        vals.append((method, val))

    vals.sort(key=lambda x: x[1], reverse=higher_is_better)

    styles = {m: "" for m in method_order}

    if len(vals) >= 1:
        styles[vals[0][0]] = "best"
    if len(vals) >= 2:
        styles[vals[1][0]] = "second"

    return styles


def build_main_table(agg: Dict[str, AggMetrics]) -> str:
    method_order = [m for m in MAIN_METHOD_ORDER if m in agg]

    best_styles = rank_methods_for_metric(
        agg, method_order, "best_val_mean", higher_is_better=True
    )
    final_styles = rank_methods_for_metric(
        agg, method_order, "final_val_mean", higher_is_better=True
    )
    drop_styles = rank_methods_for_metric(
        agg, method_order, "drop_val_mean", higher_is_better=False
    )

    lines: List[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{Method} & \\multirow{2}{*}{$n$} "
        "& \\multicolumn{4}{c}{Validation} \\\\"
    )
    lines.append("& & Best (\\%) & Final (\\%) & Drop (pp) & Peak epoch \\\\")
    lines.append("\\midrule")

    for method in method_order:
        a = agg[method]

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
        peak_cell = format_pm_float(a.peak_epoch_mean, a.peak_epoch_std)

        lines.append(
            f"{method} & {a.n} & {best_cell} & {final_cell} & "
            f"{drop_cell} & {peak_cell} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Clean CIFAR-100 results. Best and final validation accuracy are reported as "
        "mean $\\pm$ standard deviation across seeds. Drop denotes the best-to-final validation "
        "decrease in percentage points, where lower is better. Bold and italics mark the best "
        "and second-best values, respectively.}"
    )
    lines.append("\\label{tab:cifar_clean_main}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_ablation_table(agg: Dict[str, AggMetrics]) -> str:
    method_order = [m for m in ABLATION_METHOD_ORDER if m in agg]

    best_styles = rank_methods_for_metric(
        agg, method_order, "best_val_mean", higher_is_better=True
    )
    final_styles = rank_methods_for_metric(
        agg, method_order, "final_val_mean", higher_is_better=True
    )
    drop_styles = rank_methods_for_metric(
        agg, method_order, "drop_val_mean", higher_is_better=False
    )

    lines: List[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{Method} & \\multirow{2}{*}{$n$} "
        "& \\multicolumn{4}{c}{Validation} \\\\"
    )
    lines.append("& & Best (\\%) & Final (\\%) & Drop (pp) & Peak epoch \\\\")
    lines.append("\\midrule")

    for method in method_order:
        a = agg[method]

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
        peak_cell = format_pm_float(a.peak_epoch_mean, a.peak_epoch_std)

        lines.append(
            f"{method} & {a.n} & {best_cell} & {final_cell} & "
            f"{drop_cell} & {peak_cell} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Fixed learning-rate multiplier ablation on clean CIFAR-100. "
        "The fixed multipliers use the same LR multiplier values available to the AEES controller. "
        "Fixed 1.0 corresponds to the AdamW baseline. This ablation tests whether AEES improves "
        "beyond choosing a single static LR multiplier. Best and final validation accuracy are reported "
        "as mean $\\pm$ standard deviation across seeds. Drop denotes the best-to-final validation "
        "decrease in percentage points, where lower is better. Bold and italics mark the best and "
        "second-best values, respectively.}"
    )
    lines.append("\\label{tab:cifar_clean_fixed_lr_ablation}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def build_combined_compact_table(agg: Dict[str, AggMetrics]) -> str:
    method_order = [
        "AdamW",
        "Fixed 0.5",
        "Fixed 2.0",
        "AEES",
        "AdamW + Cosine",
        "AEES + Cosine",
        "AdamW + Linear",
        "AEES + Linear",
    ]
    method_order = [m for m in method_order if m in agg]

    best_styles = rank_methods_for_metric(
        agg, method_order, "best_val_mean", higher_is_better=True
    )
    final_styles = rank_methods_for_metric(
        agg, method_order, "final_val_mean", higher_is_better=True
    )
    drop_styles = rank_methods_for_metric(
        agg, method_order, "drop_val_mean", higher_is_better=False
    )

    lines: List[str] = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4.5pt}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\multirow{2}{*}{Method} & \\multirow{2}{*}{$n$} "
        "& \\multicolumn{4}{c}{Validation} \\\\"
    )
    lines.append("& & Best (\\%) & Final (\\%) & Drop (pp) & Peak epoch \\\\")
    lines.append("\\midrule")

    for method in method_order:
        a = agg[method]

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
        peak_cell = format_pm_float(a.peak_epoch_mean, a.peak_epoch_std)

        lines.append(
            f"{method} & {a.n} & {best_cell} & {final_cell} & "
            f"{drop_cell} & {peak_cell} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Compact clean CIFAR-100 table including main methods and fixed LR multiplier ablations. "
        "Fixed multipliers use the same LR multiplier values available to AEES.}"
    )
    lines.append("\\label{tab:cifar_clean_compact_appendix}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def write_csv(agg: Dict[str, AggMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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

        for method in sorted(agg.keys()):
            a = agg[method]
            writer.writerow([
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
            f"No matching clean CIFAR JSON files found in {RESULTS_ROOT}")

    print_seed_coverage(runs)

    agg = aggregate_runs(runs)

    main_table = build_main_table(agg)
    main_path = OUTPUT_DIR / "cifar_clean_main_table.tex"
    main_path.write_text(main_table)
    print(f"Wrote {main_path}")

    ablation_table = build_ablation_table(agg)
    ablation_path = OUTPUT_DIR / "cifar_clean_fixed_lr_ablation_table.tex"
    ablation_path.write_text(ablation_table)
    print(f"Wrote {ablation_path}")

    compact_table = build_combined_compact_table(agg)
    compact_path = OUTPUT_DIR / "cifar_clean_compact_appendix_table.tex"
    compact_path.write_text(compact_table)
    print(f"Wrote {compact_path}")

    all_tables = "\n\n".join([main_table, ablation_table, compact_table])
    all_tables_path = OUTPUT_DIR / "cifar_clean_tables_all.tex"
    all_tables_path.write_text(all_tables)
    print(f"Wrote {all_tables_path}")

    csv_path = OUTPUT_DIR / "cifar_clean_summary.csv"
    write_csv(agg, csv_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
