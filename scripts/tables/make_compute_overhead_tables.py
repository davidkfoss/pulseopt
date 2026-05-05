#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


RESULTS_DIR = Path("results/compute_overhead")
OUTPUT_DIR = Path("results/tables/compute_overhead")

TASK_ORDER = ["CIFAR-100", "SST-2", "AG News"]
OPTIMIZER_ORDER = ["SGD", "AdamW"]

CONFIG_ORDER = [
    "baseline_none",
    "baseline_cosine",
    "baseline_warmup_linear",
    "aees_lr_only_none",
    "aees_noise_only_none",
    "aees_both_none",
    "aees_lr_only_cosine",
    "aees_noise_only_cosine",
    "aees_both_cosine",
    "aees_lr_only_warmup_linear",
    "aees_noise_only_warmup_linear",
    "aees_both_warmup_linear",
]

CONFIG_LABELS = {
    "baseline_none": "Baseline",
    "baseline_cosine": "Cosine",
    "baseline_warmup_linear": "Warmup-linear",
    "aees_lr_only_none": r"\method{}-LR",
    "aees_noise_only_none": r"\method{}-Noise",
    "aees_both_none": r"\method{}-Dual",
    "aees_lr_only_cosine": r"Cosine + \method{}-LR",
    "aees_noise_only_cosine": r"Cosine + \method{}-Noise",
    "aees_both_cosine": r"Cosine + \method{}-Dual",
    "aees_lr_only_warmup_linear": r"Warmup-linear + \method{}-LR",
    "aees_noise_only_warmup_linear": r"Warmup-linear + \method{}-Noise",
    "aees_both_warmup_linear": r"Warmup-linear + \method{}-Dual",
}


@dataclass(frozen=True)
class Run:
    path: Path
    task: str
    optimizer: str
    config: str
    run_id: str
    wall_minutes: float
    sec_per_epoch: float
    sec_per_step: float
    steps_per_second: float
    tflops_per_second: float
    total_epochs: int
    total_steps: int


@dataclass(frozen=True)
class AggregateRun:
    task: str
    optimizer: str
    config: str
    n: int
    paths: list[Path]

    wall_minutes_mean: float
    wall_minutes_std: float
    sec_per_epoch_mean: float
    sec_per_epoch_std: float
    sec_per_step_mean: float
    sec_per_step_std: float
    steps_per_second_mean: float
    steps_per_second_std: float
    tflops_per_second_mean: float
    tflops_per_second_std: float

    total_epochs: int
    total_steps: int


def escape_latex(s: str) -> str:
    if "\\" in s:
        return s
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def strip_repeat_suffix(stem: str) -> tuple[str, str]:
    """
    Converts:
      foo_seed0_run2 -> (foo_seed0, run2)
      foo_seed0_run3 -> (foo_seed0, run3)
      foo_seed0      -> (foo_seed0, run1)
    """
    match = re.search(r"_run(\d+)$", stem)
    if match:
        return stem[: match.start()], f"run{match.group(1)}"
    return stem, "run1"


def canonical_config_name(stem: str) -> str:
    canonical_stem, _ = strip_repeat_suffix(stem)
    config = canonical_stem

    prefixes = [
        "cifar_clean_sgd_",
        "cifar_clean_adamw_",
        "sst2_clean_adamw_",
        "agnews_clean_adamw_",
    ]
    for prefix in prefixes:
        if config.startswith(prefix):
            config = config[len(prefix):]
            break

    config = re.sub(r"_seed\d+$", "", config)
    return config


def infer_task_optimizer_config(path: Path, data: dict) -> tuple[str, str, str, str]:
    stem = path.stem
    lower = stem.lower()
    canonical_stem, run_id = strip_repeat_suffix(stem)

    if canonical_stem.startswith("cifar_"):
        task = "CIFAR-100"
    elif canonical_stem.startswith("sst2_"):
        task = "SST-2"
    elif canonical_stem.startswith("agnews_"):
        task = "AG News"
    else:
        task = str(data.get("task_name") or data.get(
            "dataset_name") or "Unknown")

    if "_sgd_" in lower or lower.startswith("cifar_clean_sgd"):
        optimizer = "SGD"
    elif "_adamw_" in lower:
        optimizer = "AdamW"
    elif data.get("method_name") == "AdamW" or data.get("config", {}).get("method") == "AdamW":
        optimizer = "AdamW"
    elif data.get("config", {}).get("optimizer") == "AdamW":
        optimizer = "AdamW"
    else:
        optimizer = str(
            data.get("config", {}).get("optimizer")
            or data.get("config", {}).get("method")
            or "Unknown"
        )

    config = canonical_config_name(stem)
    return task, optimizer, config, run_id


def safe_float(x: object) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def load_run(path: Path) -> Optional[Run]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    runtime = data.get("runtime_metrics", {})
    if not runtime:
        return None

    task, optimizer, config, run_id = infer_task_optimizer_config(path, data)

    wall_seconds = safe_float(runtime.get(
        "wall_clock_seconds") or data.get("wall_clock_seconds"))
    total_epochs = int(data.get("total_epochs")
                       or runtime.get("total_epochs") or 0)
    total_steps = int(data.get("total_steps")
                      or runtime.get("total_steps") or 0)

    epoch_times = data.get("epoch_wall_clock_seconds") or runtime.get(
        "epoch_wall_clock_seconds")

    if isinstance(epoch_times, list) and len(epoch_times) > 1:
        clean_epoch_times = [safe_float(x) for x in epoch_times[1:]]
        clean_epoch_times = [x for x in clean_epoch_times if not math.isnan(x)]
        sec_per_epoch = mean(
            clean_epoch_times) if clean_epoch_times else math.nan
    else:
        sec_per_epoch = safe_float(
            runtime.get("seconds_per_epoch")
            or runtime.get("mean_epoch_wall_clock_seconds")
            or (wall_seconds / max(1, total_epochs) if not math.isnan(wall_seconds) else math.nan)
        )

    return Run(
        path=path,
        task=task,
        optimizer=optimizer,
        config=config,
        run_id=run_id,
        wall_minutes=wall_seconds /
        60.0 if not math.isnan(wall_seconds) else math.nan,
        sec_per_epoch=sec_per_epoch,
        sec_per_step=safe_float(runtime.get("seconds_per_step")),
        steps_per_second=safe_float(runtime.get("steps_per_second")),
        tflops_per_second=safe_float(
            runtime.get("estimated_tflops_per_second")),
        total_epochs=total_epochs,
        total_steps=total_steps,
    )


def collect_runs() -> list[Run]:
    runs: list[Run] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        run = load_run(path)
        if run is not None:
            runs.append(run)
    return runs


def clean_values(values: list[float]) -> list[float]:
    return [v for v in values if v is not None and not math.isnan(v)]


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = clean_values(values)
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], math.nan
    return mean(vals), stdev(vals)


def aggregate_runs(runs: list[Run]) -> list[AggregateRun]:
    grouped: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault(
            (run.task, run.optimizer, run.config), []).append(run)

    aggregates: list[AggregateRun] = []

    for (task, optimizer, config), group_runs in grouped.items():
        group_runs = sorted(group_runs, key=lambda r: r.run_id)

        wall_m, wall_s = mean_std([r.wall_minutes for r in group_runs])
        spe_m, spe_s = mean_std([r.sec_per_epoch for r in group_runs])
        sps_m, sps_s = mean_std([r.sec_per_step for r in group_runs])
        eps_m, eps_s = mean_std([r.steps_per_second for r in group_runs])
        tflops_m, tflops_s = mean_std(
            [r.tflops_per_second for r in group_runs])

        aggregates.append(
            AggregateRun(
                task=task,
                optimizer=optimizer,
                config=config,
                n=len(group_runs),
                paths=[r.path for r in group_runs],
                wall_minutes_mean=wall_m,
                wall_minutes_std=wall_s,
                sec_per_epoch_mean=spe_m,
                sec_per_epoch_std=spe_s,
                sec_per_step_mean=sps_m,
                sec_per_step_std=sps_s,
                steps_per_second_mean=eps_m,
                steps_per_second_std=eps_s,
                tflops_per_second_mean=tflops_m,
                tflops_per_second_std=tflops_s,
                total_epochs=group_runs[0].total_epochs,
                total_steps=group_runs[0].total_steps,
            )
        )

    return sorted(aggregates, key=sort_key)


def sort_key(run: AggregateRun | Run) -> tuple[int, int, int, str]:
    task_idx = TASK_ORDER.index(run.task) if run.task in TASK_ORDER else 999
    opt_idx = OPTIMIZER_ORDER.index(
        run.optimizer) if run.optimizer in OPTIMIZER_ORDER else 999
    config_idx = CONFIG_ORDER.index(
        run.config) if run.config in CONFIG_ORDER else 999
    return task_idx, opt_idx, config_idx, run.config


def group_baselines(aggregates: list[AggregateRun]) -> dict[tuple[str, str], AggregateRun]:
    return {
        (agg.task, agg.optimizer): agg
        for agg in aggregates
        if agg.config == "baseline_none"
    }


def overhead_vs_baseline(
    run: Optional[AggregateRun],
    baselines: dict[tuple[str, str], AggregateRun],
) -> Optional[float]:
    if run is None:
        return None
    base = baselines.get((run.task, run.optimizer))
    if base is None or math.isnan(base.sec_per_epoch_mean) or base.sec_per_epoch_mean <= 0:
        return None
    return 100.0 * (run.sec_per_epoch_mean / base.sec_per_epoch_mean - 1.0)


def fmt_float(x: Optional[float], digits: int = 2) -> str:
    if x is None or math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_pm(mean_val: Optional[float], std_val: Optional[float], digits: int = 2) -> str:
    if mean_val is None or math.isnan(mean_val):
        return "--"
    if std_val is None or math.isnan(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} $\\pm$ {std_val:.{digits}f}"


def fmt_overhead(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "--"
    return f"{x:+.1f}"


def main_scheduler_config(task: str) -> str:
    return "baseline_cosine" if task == "CIFAR-100" else "baseline_warmup_linear"


def main_table_configs(task: str) -> list[str]:
    return [
        main_scheduler_config(task),
        "aees_lr_only_none",
        "aees_noise_only_none",
        "aees_both_none",
    ]


def build_main_overhead_table(aggregates: list[AggregateRun]) -> str:
    agg_map = {(a.task, a.optimizer, a.config): a for a in aggregates}
    baselines = group_baselines(aggregates)

    groups = sorted(
        {(a.task, a.optimizer)
         for a in aggregates if (a.task, a.optimizer) in baselines},
        key=lambda x: (
            TASK_ORDER.index(x[0]) if x[0] in TASK_ORDER else 999,
            OPTIMIZER_ORDER.index(x[1]) if x[1] in OPTIMIZER_ORDER else 999,
        ),
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{")
    lines.append(
        r"    Wall-clock training overhead of scheduler and \method{} configurations, measured on a single NVIDIA RTX PRO 6000.")
    lines.append(
        r"    Values are averaged over repeated timing runs with the same random seed.")
    lines.append(
        r"    The \emph{Baseline} column reports seconds per epoch for the corresponding no-scheduler optimizer baseline;")
    lines.append(
        r"    all remaining columns report percentage change in epoch wall-clock time relative to that baseline.")
    lines.append(r"    \method{}-LR adapts only the learning-rate multiplier, \method{}-Noise adapts only gradient-noise magnitude, and \method{}-Dual adapts both axes.")
    lines.append(r"  }")
    lines.append(r"  \label{tab:compute-overhead}")
    lines.append(r"  \sisetup{")
    lines.append(r"    table-format = +2.1,")
    lines.append(r"    table-space-text-post = {\,\%},")
    lines.append(r"    detect-weight,")
    lines.append(r"    round-mode = places,")
    lines.append(r"    round-precision = 1")
    lines.append(r"  }")
    lines.append(r"  \begin{tabular}{")
    lines.append(r"      l")
    lines.append(r"      S[table-format=3.2]")
    lines.append(r"      S[table-format=+2.1]")
    lines.append(r"      S[table-format=+2.1]")
    lines.append(r"      S[table-format=+2.1]")
    lines.append(r"      S[table-format=+2.1]")
    lines.append(r"    }")
    lines.append(r"    \toprule")
    lines.append(
        r"    & & \multicolumn{4}{c}{Overhead relative to baseline (\%)} \\")
    lines.append(r"    \cmidrule(l){3-6}")
    lines.append(r"    {Task / Optimizer}")
    lines.append(r"      & {Baseline (s/ep)}")
    lines.append(r"      & {LR sched.}")
    lines.append(r"      & {\method{}-LR}")
    lines.append(r"      & {\method{}-Noise}")
    lines.append(r"      & {\method{}-Dual} \\")
    lines.append(r"    \midrule")

    for task, optimizer in groups:
        base = baselines[(task, optimizer)]
        config_values = []
        for config in main_table_configs(task):
            agg = agg_map.get((task, optimizer, config))
            config_values.append(fmt_overhead(
                overhead_vs_baseline(agg, baselines)))

        lines.append(
            f"    {escape_latex(task)} / {escape_latex(optimizer)}"
            f" & {fmt_float(base.sec_per_epoch_mean, 2)}"
            f" & {config_values[0]} & {config_values[1]} & {config_values[2]} & {config_values[3]} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_detailed_table(aggregates: list[AggregateRun]) -> str:
    baselines = group_baselines(aggregates)

    lines: list[str] = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lllrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Task & Optimizer & Configuration & $n$ & Time/epoch & Overhead & Steps/s & TFLOP/s \\"
    )
    lines.append(
        r"& & & & (s) & (\%) & & \\"
    )
    lines.append(r"\midrule")

    prev_group: Optional[tuple[str, str]] = None

    for agg in aggregates:
        group = (agg.task, agg.optimizer)
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        label = CONFIG_LABELS.get(agg.config, agg.config)
        overhead = overhead_vs_baseline(agg, baselines)

        lines.append(
            f"{escape_latex(agg.task)} & "
            f"{escape_latex(agg.optimizer)} & "
            f"{escape_latex(label)} & "
            f"{agg.n} & "
            f"{fmt_pm(agg.sec_per_epoch_mean, agg.sec_per_epoch_std, 2)} & "
            f"{fmt_overhead(overhead)} & "
            f"{fmt_pm(agg.steps_per_second_mean, agg.steps_per_second_std, 2)} & "
            f"{fmt_pm(agg.tflops_per_second_mean, agg.tflops_per_second_std, 2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Detailed per-configuration wall-clock and throughput measurements for the compute-overhead experiment. "
        r"Values are reported as mean $\pm$ standard deviation across repeated timing runs with the same random seed. "
        r"Overhead is measured relative to the corresponding no-scheduler baseline within the same task and optimizer setting. "
        r"This table is supplementary runtime evidence rather than a basis for averaging across heterogeneous \method{} variants.}"
    )
    lines.append(r"\label{tab:compute_overhead_detailed}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def write_csv(aggregates: list[AggregateRun]) -> None:
    baselines = group_baselines(aggregates)
    out_path = OUTPUT_DIR / "compute_overhead_summary.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "optimizer",
            "config",
            "label",
            "n",
            "wall_minutes_mean",
            "wall_minutes_std",
            "sec_per_epoch_mean",
            "sec_per_epoch_std",
            "sec_per_step_mean",
            "sec_per_step_std",
            "steps_per_second_mean",
            "steps_per_second_std",
            "tflops_per_second_mean",
            "tflops_per_second_std",
            "overhead_vs_baseline_none_pct",
            "total_epochs",
            "total_steps",
            "paths",
        ])

        for agg in sorted(aggregates, key=sort_key):
            writer.writerow([
                agg.task,
                agg.optimizer,
                agg.config,
                CONFIG_LABELS.get(agg.config, agg.config),
                agg.n,
                agg.wall_minutes_mean,
                agg.wall_minutes_std,
                agg.sec_per_epoch_mean,
                agg.sec_per_epoch_std,
                agg.sec_per_step_mean,
                agg.sec_per_step_std,
                agg.steps_per_second_mean,
                agg.steps_per_second_std,
                agg.tflops_per_second_mean,
                agg.tflops_per_second_std,
                overhead_vs_baseline(agg, baselines),
                agg.total_epochs,
                agg.total_steps,
                ";".join(str(p) for p in agg.paths),
            ])

    print(f"Wrote {out_path}")


def print_coverage(runs: list[Run], aggregates: list[AggregateRun]) -> None:
    found = {(a.task, a.optimizer, a.config): a for a in aggregates}

    print("\n==============================")
    print("COMPUTE OVERHEAD COVERAGE")
    print("==============================")

    for task in TASK_ORDER:
        task_opts = sorted(
            {a.optimizer for a in aggregates if a.task == task},
            key=lambda x: OPTIMIZER_ORDER.index(
                x) if x in OPTIMIZER_ORDER else 999,
        )
        if not task_opts:
            continue

        print(f"\n[{task}]")
        for optimizer in task_opts:
            print(f"  {optimizer}")
            for config in CONFIG_ORDER:
                agg = found.get((task, optimizer, config))
                if agg is not None:
                    run_names = ", ".join(p.name for p in agg.paths)
                    print(f"    {config:<35} OK  n={agg.n}  ({run_names})")

    print("\nRaw run count:", len(runs))
    print("Aggregated condition count:", len(aggregates))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs()
    if not runs:
        raise SystemExit(f"No JSON result files found in {RESULTS_DIR}")

    aggregates = aggregate_runs(runs)
    print_coverage(runs, aggregates)

    main_table = build_main_overhead_table(aggregates)
    main_path = OUTPUT_DIR / "compute_overhead_decomposition_table.tex"
    main_path.write_text(main_table)
    print(f"Wrote {main_path}")

    detailed = build_detailed_table(aggregates)
    detailed_path = OUTPUT_DIR / "compute_overhead_detailed_table.tex"
    detailed_path.write_text(detailed)
    print(f"Wrote {detailed_path}")

    combined_path = OUTPUT_DIR / "compute_overhead_tables_all.tex"
    combined_path.write_text(main_table + "\n\n" + detailed)
    print(f"Wrote {combined_path}")

    write_csv(aggregates)


if __name__ == "__main__":
    main()
