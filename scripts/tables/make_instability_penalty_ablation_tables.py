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

RESULTS_ROOT = Path("results/instability_penalty_ablation")
OUTPUT_DIR = Path("results/tables/ablations")

SETTING_ORDER = ["cifar_clean", "cifar_noisy"]
SETTING_LABELS = {
    "cifar_clean": "Clean \\cifar{}",
    "cifar_noisy": "Sym. 20\\% \\cifar{}",
}

VARIANT_ORDER = ["0p0", "0p1"]
VARIANT_LABELS = {
    "0p0": "No penalty",
    "0p1": "Penalty",
}

# Expected file names under seed folders:
# seed_0/clean_penalty_0p0_seed0.json
# seed_0/clean_penalty_0p1_seed0.json
# seed_0/noisy_penalty_0p0_seed0.json
# seed_0/noisy_penalty_0p1_seed0.json

FILE_RE = re.compile(r"^(clean|noisy)_penalty_(0p0|0p1)_seed(\d+)\.json$")


@dataclass
class RunMetrics:
    setting: str
    variant: str
    seed: int
    best_val: float
    final_val: float
    drop_val: float
    path: Path


@dataclass
class AggMetrics:
    n: int
    best_mean: float
    best_std: float
    final_mean: float
    final_std: float
    drop_mean: float
    drop_std: float


def safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else math.nan


def safe_std(xs: List[float]) -> float:
    if not xs:
        return math.nan
    return stdev(xs) if len(xs) > 1 else 0.0


def pct(x: float) -> float:
    return 100.0 * x


def fmt_pm_pct(mean_val: float, std_val: float) -> str:
    if math.isnan(mean_val):
        return "--"
    if math.isnan(std_val):
        return f"{pct(mean_val):.1f}"
    return f"{pct(mean_val):.1f} $\\pm$ {pct(std_val):.1f}"


def fmt_delta_pct(delta: float) -> str:
    if math.isnan(delta):
        return "--"
    val = pct(delta)
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f}"


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
    if not path.is_file() or path.suffix != ".json":
        return None

    match = FILE_RE.match(path.name)
    if not match:
        return None

    dataset, variant, seed_raw = match.groups()
    setting = "cifar_clean" if dataset == "clean" else "cifar_noisy"
    seed = int(seed_raw)

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    val_accuracies = get_val_accuracies(data)
    if not val_accuracies:
        return None

    best_val = get_best_val(data, val_accuracies)
    final_val = get_final_val(data, val_accuracies)

    return RunMetrics(
        setting=setting,
        variant=variant,
        seed=seed,
        best_val=best_val,
        final_val=final_val,
        drop_val=best_val - final_val,
        path=path,
    )


def collect_runs(root: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []

    for path in root.glob("seed_*/*.json"):
        run = load_run(path)
        if run is not None:
            runs.append(run)

    return runs


def aggregate_runs(runs: List[RunMetrics]) -> Dict[Tuple[str, str], AggMetrics]:
    grouped: Dict[Tuple[str, str], List[RunMetrics]] = defaultdict(list)

    for run in runs:
        grouped[(run.setting, run.variant)].append(run)

    out: Dict[Tuple[str, str], AggMetrics] = {}

    for key, group in grouped.items():
        best_vals = [r.best_val for r in group]
        final_vals = [r.final_val for r in group]
        drop_vals = [r.drop_val for r in group]

        out[key] = AggMetrics(
            n=len(group),
            best_mean=safe_mean(best_vals),
            best_std=safe_std(best_vals),
            final_mean=safe_mean(final_vals),
            final_std=safe_std(final_vals),
            drop_mean=safe_mean(drop_vals),
            drop_std=safe_std(drop_vals),
        )

    return out


def print_coverage(runs: List[RunMetrics]) -> None:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for r in runs:
        grouped[(r.setting, r.variant)].append(r.seed)

    print("\n====================================")
    print("INSTABILITY PENALTY ABLATION COVERAGE")
    print("====================================")

    for setting in SETTING_ORDER:
        print(f"\n[{setting}]")
        for variant in VARIANT_ORDER:
            seeds = sorted(set(grouped.get((setting, variant), [])))
            if seeds:
                print(f"  {variant:<4} n={len(seeds)} seeds={seeds}")
            else:
                print(f"  {variant:<4} MISSING")


def get_agg(
    agg: Dict[Tuple[str, str], AggMetrics],
    setting: str,
    variant: str,
) -> Optional[AggMetrics]:
    return agg.get((setting, variant))


def build_table(agg: Dict[Tuple[str, str], AggMetrics]) -> str:
    lines: List[str] = []

    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption[Instability-penalty reward ablation]{")
    lines.append(
        "Instability-penalty reward ablation. The penalty variant uses the normalized within-episode "
        "loss-variance term from \\Cref{sec:reward-design}. Values are reported as mean $\\pm$ "
        "standard deviation across seeds. Differences are computed as penalty minus no penalty."
    )
    lines.append("}")
    lines.append("\\label{tab:instability-penalty-ablation}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lccccccccc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{Setting} &")
    lines.append("\\multicolumn{3}{c}{No penalty} &")
    lines.append("\\multicolumn{3}{c}{Penalty} &")
    lines.append("\\multicolumn{3}{c}{Difference} \\\\")
    lines.append("\\cmidrule(lr){2-4}")
    lines.append("\\cmidrule(lr){5-7}")
    lines.append("\\cmidrule(lr){8-10}")
    lines.append(
        "& $\\accbest$ & $\\accfinal$ & $\\dropmetric$ & "
        "$\\accbest$ & $\\accfinal$ & $\\dropmetric$ & "
        "$\\Delta\\accbest$ & $\\Delta\\accfinal$ & $\\Delta\\dropmetric$ \\\\"
    )
    lines.append("\\midrule")

    for setting in SETTING_ORDER:
        none = get_agg(agg, setting, "0p0")
        penalty = get_agg(agg, setting, "0p1")

        if none is None or penalty is None:
            none_best = none_final = none_drop = "--"
            penalty_best = penalty_final = penalty_drop = "--"
            d_best = d_final = d_drop = "--"
        else:
            none_best = fmt_pm_pct(none.best_mean, none.best_std)
            none_final = fmt_pm_pct(none.final_mean, none.final_std)
            none_drop = fmt_pm_pct(none.drop_mean, none.drop_std)

            penalty_best = fmt_pm_pct(penalty.best_mean, penalty.best_std)
            penalty_final = fmt_pm_pct(penalty.final_mean, penalty.final_std)
            penalty_drop = fmt_pm_pct(penalty.drop_mean, penalty.drop_std)

            d_best = fmt_delta_pct(penalty.best_mean - none.best_mean)
            d_final = fmt_delta_pct(penalty.final_mean - none.final_mean)
            d_drop = fmt_delta_pct(penalty.drop_mean - none.drop_mean)

        lines.append(
            f"{SETTING_LABELS[setting]} & "
            f"{none_best} & {none_final} & {none_drop} & "
            f"{penalty_best} & {penalty_final} & {penalty_drop} & "
            f"{d_best} & {d_final} & {d_drop} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def write_csv(
    agg: Dict[Tuple[str, str], AggMetrics],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "setting",
            "variant",
            "n",
            "best_mean",
            "best_std",
            "final_mean",
            "final_std",
            "drop_mean",
            "drop_std",
        ])

        for setting in SETTING_ORDER:
            for variant in VARIANT_ORDER:
                a = agg.get((setting, variant))
                if a is None:
                    continue
                writer.writerow([
                    setting,
                    variant,
                    a.n,
                    a.best_mean,
                    a.best_std,
                    a.final_mean,
                    a.final_std,
                    a.drop_mean,
                    a.drop_std,
                ])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(RESULTS_ROOT)
    if not runs:
        raise SystemExit(
            f"No matching instability-penalty ablation JSON files found in {RESULTS_ROOT}")

    print_coverage(runs)

    agg = aggregate_runs(runs)

    table = build_table(agg)
    table_path = OUTPUT_DIR / "instability_penalty_ablation_table.tex"
    table_path.write_text(table)
    print(f"Wrote {table_path}")

    csv_path = OUTPUT_DIR / "instability_penalty_ablation_summary.csv"
    write_csv(agg, csv_path)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
