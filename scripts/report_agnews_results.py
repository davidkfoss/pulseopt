#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_RESULT_DIRS = [
    "results/agnews_final_warmup_linear",
    "results/agnews_final_none",
    "results/agnews_final_warmup_linear_coarse_highnoise",
]


@dataclass(frozen=True)
class RunSummary:
    path: Path
    seed: int
    label: str
    method: str
    scheduler: str
    context_mode: str
    episode_length: int | None
    lr_candidates: tuple[float, ...]
    noise_candidates: tuple[float, ...]
    best_val_accuracy: float
    final_val_accuracy: float
    best_epoch: int
    final_train_loss: float | None
    total_epochs: int
    total_steps: int | None
    total_tflops: float | None
    tflops_to_best: float | None
    total_time: float | None
    time_to_best: float | None
    best_acc_per_total_tflop: float | None
    final_acc_per_total_tflop: float | None
    best_acc_per_tflop_to_best: float | None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def as_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def get_config(data: dict[str, Any]) -> dict[str, Any]:
    cfg = data.get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def parse_seed(data: dict[str, Any], path: Path) -> int:
    if "seed" in data:
        return int(data["seed"])
    cfg = get_config(data)
    if "seed" in cfg:
        return int(cfg["seed"])
    match = re.search(r"seed(\d+)", path.name)
    if match:
        return int(match.group(1))
    return -1


def parse_float_tuple(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple()
    if isinstance(value, list):
        return tuple(float(x) for x in value)
    if isinstance(value, tuple):
        return tuple(float(x) for x in value)
    return tuple()


def infer_label(data: dict[str, Any], path: Path) -> str:
    cfg = get_config(data)

    method = str(data.get("method_name", cfg.get("method", "unknown")))
    scheduler = str(cfg.get("lr_scheduler", "none"))
    context = str(cfg.get("context_mode", "none"))

    lr_candidates = parse_float_tuple(cfg.get("lr_candidates"))
    noise_candidates = parse_float_tuple(cfg.get("noise_candidates"))
    episode_length = cfg.get("episode_length")

    if method == "AdamW":
        if scheduler == "none":
            return "AdamW"
        return f"AdamW + {scheduler}"

    if method == "AdaptiveScheduler":
        lr_part = "LR"
        noise_part = "noise"

        if lr_candidates in [tuple(), (1.0,)]:
            lr_part = "fixed-LR"
        if noise_candidates in [tuple(), (0.0,)]:
            noise_part = "no-noise"

        grid_name = ""
        if lr_candidates == (0.7, 1.0, 1.3) and noise_candidates == (0.0, 0.0025, 0.005):
            grid_name = "SST2-grid"
        elif lr_candidates == (0.5, 1.0, 2.0) and noise_candidates == (0.0, 0.005, 0.01):
            grid_name = "coarse/high-noise"
        elif lr_candidates or noise_candidates:
            grid_name = f"LR={lr_candidates}, noise={noise_candidates}"

        sched_part = "" if scheduler == "none" else f" + {scheduler}"
        ctx_part = "" if context == "none" else f" + ctx={context}"
        ep_part = "" if episode_length is None else f", ep={episode_length}"

        return f"AEES{sched_part} ({grid_name}{ep_part}{ctx_part})"

    return path.stem


def summarize_run(path: Path) -> RunSummary | None:
    data = load_json(path)
    cfg = get_config(data)

    task_name = str(data.get("task_name", cfg.get("task_name", ""))).lower()
    dataset_name = str(
        data.get("dataset_name", cfg.get("dataset_name", ""))).lower()
    if "agnews" not in task_name and "ag news" not in dataset_name and "ag_news" not in str(path):
        return None

    val_accuracies = as_float_list(data.get("val_accuracies"))
    if not val_accuracies:
        best_val_accuracy = float(data.get("best_val_accuracy"))
        final_val_accuracy = float(data.get("final_val_accuracy"))
        best_epoch = -1
    else:
        best_epoch_zero_indexed = max(
            range(len(val_accuracies)), key=lambda i: val_accuracies[i])
        best_epoch = best_epoch_zero_indexed + 1
        best_val_accuracy = float(
            data.get("best_val_accuracy", val_accuracies[best_epoch_zero_indexed]))
        final_val_accuracy = float(
            data.get("final_val_accuracy", val_accuracies[-1]))

    epoch_tflops = as_float_list(data.get("epoch_tflops"))
    epoch_times = as_float_list(data.get("epoch_wall_clock_seconds"))

    total_tflops = data.get("estimated_flops")
    if total_tflops is not None:
        total_tflops = float(total_tflops) / 1e12
    elif epoch_tflops:
        total_tflops = sum(epoch_tflops)
    else:
        total_tflops = None

    tflops_to_best = None
    if epoch_tflops and best_epoch > 0:
        tflops_to_best = sum(epoch_tflops[:best_epoch])

    total_time = data.get("wall_clock_seconds")
    if total_time is not None:
        total_time = float(total_time)
    elif epoch_times:
        total_time = sum(epoch_times)
    else:
        total_time = None

    time_to_best = None
    if epoch_times and best_epoch > 0:
        time_to_best = sum(epoch_times[:best_epoch])

    def safe_div(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b <= 0:
            return None
        return a / b

    return RunSummary(
        path=path,
        seed=parse_seed(data, path),
        label=infer_label(data, path),
        method=str(data.get("method_name", cfg.get("method", "unknown"))),
        scheduler=str(cfg.get("lr_scheduler", "none")),
        context_mode=str(cfg.get("context_mode", "none")),
        episode_length=int(cfg["episode_length"]) if cfg.get(
            "episode_length") is not None else None,
        lr_candidates=parse_float_tuple(cfg.get("lr_candidates")),
        noise_candidates=parse_float_tuple(cfg.get("noise_candidates")),
        best_val_accuracy=best_val_accuracy,
        final_val_accuracy=final_val_accuracy,
        best_epoch=best_epoch,
        final_train_loss=float(data["final_train_loss"]) if data.get(
            "final_train_loss") is not None else None,
        total_epochs=int(data.get("total_epochs", cfg.get("epochs", -1))),
        total_steps=int(data["total_steps"]) if data.get(
            "total_steps") is not None else None,
        total_tflops=total_tflops,
        tflops_to_best=tflops_to_best,
        total_time=total_time,
        time_to_best=time_to_best,
        best_acc_per_total_tflop=safe_div(best_val_accuracy, total_tflops),
        final_acc_per_total_tflop=safe_div(final_val_accuracy, total_tflops),
        best_acc_per_tflop_to_best=safe_div(best_val_accuracy, tflops_to_best),
    )


def fmt_mean_std(values: list[float], percent: bool = False, digits: int = 2) -> str:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "n/a"
    scale = 100.0 if percent else 1.0
    m = mean(values) * scale
    s = stdev(values) * scale if len(values) > 1 else 0.0
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def fmt_mean(values: list[float], digits: int = 2) -> str:
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "n/a"
    return f"{mean(values):.{digits}f}"


def collect(paths: list[Path]) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for root in paths:
        if root.is_file() and root.suffix == ".json":
            run = summarize_run(root)
            if run is not None:
                runs.append(run)
            continue
        if root.exists():
            for path in sorted(root.rglob("*.json")):
                try:
                    run = summarize_run(path)
                except Exception as exc:
                    print(f"[warn] failed to parse {path}: {exc}")
                    continue
                if run is not None:
                    runs.append(run)
    return runs


def group_runs(runs: list[RunSummary]) -> dict[str, list[RunSummary]]:
    groups: dict[str, list[RunSummary]] = defaultdict(list)
    for run in runs:
        groups[run.label].append(run)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def print_detailed_runs(runs: list[RunSummary]) -> None:
    print("\nPer-run results")
    print("-" * 140)
    header = (
        f"{'seed':>4}  {'label':<55}  {'best':>8}  {'final':>8}  "
        f"{'best_ep':>7}  {'tflop_best':>10}  {'tflop_total':>11}  "
        f"{'time_best':>10}  {'time_total':>10}  path"
    )
    print(header)
    print("-" * 140)
    for r in sorted(runs, key=lambda x: (x.label, x.seed, str(x.path))):
        print(
            f"{r.seed:>4}  {r.label:<55}  "
            f"{100*r.best_val_accuracy:>7.2f}%  {100*r.final_val_accuracy:>7.2f}%  "
            f"{r.best_epoch:>7}  "
            f"{r.tflops_to_best if r.tflops_to_best is not None else float('nan'):>10.3f}  "
            f"{r.total_tflops if r.total_tflops is not None else float('nan'):>11.3f}  "
            f"{r.time_to_best if r.time_to_best is not None else float('nan'):>10.1f}  "
            f"{r.total_time if r.total_time is not None else float('nan'):>10.1f}  "
            f"{r.path}"
        )


def print_aggregate_table(groups: dict[str, list[RunSummary]]) -> None:
    print("\nAggregate AG News results")
    print("-" * 150)
    print(
        f"{'label':<58} {'n':>2} "
        f"{'best acc':>15} {'final acc':>15} {'best epoch':>12} "
        f"{'TFLOPs best':>15} {'TFLOPs total':>15} "
        f"{'time best (s)':>15} {'time total (s)':>15}"
    )
    print("-" * 150)

    for label, rs in groups.items():
        print(
            f"{label:<58} {len(rs):>2} "
            f"{fmt_mean_std([r.best_val_accuracy for r in rs], percent=True):>15} "
            f"{fmt_mean_std([r.final_val_accuracy for r in rs], percent=True):>15} "
            f"{fmt_mean_std([float(r.best_epoch) for r in rs], digits=2):>12} "
            f"{fmt_mean_std([r.tflops_to_best for r in rs if r.tflops_to_best is not None], digits=3):>15} "
            f"{fmt_mean_std([r.total_tflops for r in rs if r.total_tflops is not None], digits=3):>15} "
            f"{fmt_mean_std([r.time_to_best for r in rs if r.time_to_best is not None], digits=1):>15} "
            f"{fmt_mean_std([r.total_time for r in rs if r.total_time is not None], digits=1):>15}"
        )


def print_efficiency_table(groups: dict[str, list[RunSummary]]) -> None:
    print("\nSecondary ratio metrics")
    print("-" * 125)
    print(
        f"{'label':<58} {'n':>2} "
        f"{'best/total TFLOP':>20} {'final/total TFLOP':>20} {'best/TFLOP-to-best':>22}"
    )
    print("-" * 125)

    for label, rs in groups.items():
        print(
            f"{label:<58} {len(rs):>2} "
            f"{fmt_mean_std([r.best_acc_per_total_tflop for r in rs if r.best_acc_per_total_tflop is not None], digits=4):>20} "
            f"{fmt_mean_std([r.final_acc_per_total_tflop for r in rs if r.final_acc_per_total_tflop is not None], digits=4):>20} "
            f"{fmt_mean_std([r.best_acc_per_tflop_to_best for r in rs if r.best_acc_per_tflop_to_best is not None], digits=4):>22}"
        )


def best_label_by_metric(groups: dict[str, list[RunSummary]], attr: str, higher_is_better: bool = True) -> tuple[str, float] | None:
    scored = []
    for label, rs in groups.items():
        values = [getattr(r, attr) for r in rs]
        values = [float(v)
                  for v in values if v is not None and math.isfinite(float(v))]
        if values:
            scored.append((label, mean(values)))
    if not scored:
        return None
    return max(scored, key=lambda x: x[1]) if higher_is_better else min(scored, key=lambda x: x[1])


def print_interpretation(groups: dict[str, list[RunSummary]]) -> None:
    print("\nQuick interpretation")
    print("-" * 80)

    checks = [
        ("highest mean best validation accuracy", "best_val_accuracy", True, True),
        ("highest mean final validation accuracy",
         "final_val_accuracy", True, True),
        ("lowest mean TFLOPs to best", "tflops_to_best", False, False),
        ("lowest mean time to best", "time_to_best", False, False),
        ("lowest mean total time", "total_time", False, False),
    ]

    for desc, attr, higher, percent in checks:
        item = best_label_by_metric(groups, attr, higher_is_better=higher)
        if item is None:
            continue
        label, value = item
        value_text = f"{100*value:.2f}%" if percent else f"{value:.3f}"
        print(f"- {desc}: {label} ({value_text})")

    labels = list(groups)
    adamw = [x for x in labels if x == "AdamW"]
    adamw_sched = [x for x in labels if x.startswith("AdamW +")]
    aees = [x for x in labels if x.startswith(
        "AEES") and "warmup_linear" not in x and "linear" not in x]
    aees_sched = [x for x in labels if x.startswith(
        "AEES") and ("warmup_linear" in x or "linear" in x)]

    if adamw and aees:
        a = mean([r.best_val_accuracy for r in groups[adamw[0]]])
        b = mean([r.best_val_accuracy for r in groups[aees[0]]])
        print(f"- AEES vs AdamW best-val delta: {(b - a) * 100:+.2f} pp")

    if adamw_sched and aees_sched:
        a = mean([r.best_val_accuracy for r in groups[adamw_sched[0]]])
        b = mean([r.best_val_accuracy for r in groups[aees_sched[0]]])
        print(
            f"- AEES+scheduler vs AdamW+scheduler best-val delta: {(b - a) * 100:+.2f} pp")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def print_latex_table(groups: dict[str, list[RunSummary]]) -> None:
    print("\nLaTeX table")
    print("-" * 80)
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Method & Best acc. & Final acc. & Best epoch & TFLOPs@best & Time@best & Total time \\")
    print(r"\midrule")
    for label, rs in groups.items():
        best = fmt_mean_std([r.best_val_accuracy for r in rs], percent=True)
        final = fmt_mean_std([r.final_val_accuracy for r in rs], percent=True)
        best_epoch = fmt_mean_std([float(r.best_epoch) for r in rs], digits=2)
        tflops_best = fmt_mean_std(
            [r.tflops_to_best for r in rs if r.tflops_to_best is not None], digits=3)
        time_best = fmt_mean_std(
            [r.time_to_best for r in rs if r.time_to_best is not None], digits=1)
        time_total = fmt_mean_std(
            [r.total_time for r in rs if r.total_time is not None], digits=1)
        print(
            f"{latex_escape(label)} & {best} & {final} & {best_epoch} & "
            f"{tflops_best} & {time_best} & {time_total} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def write_csv(groups: dict[str, list[RunSummary]], output: Path) -> None:
    import csv

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "n",
            "best_val_accuracy_mean",
            "best_val_accuracy_std",
            "final_val_accuracy_mean",
            "final_val_accuracy_std",
            "best_epoch_mean",
            "tflops_to_best_mean",
            "total_tflops_mean",
            "time_to_best_mean",
            "total_time_mean",
            "best_acc_per_total_tflop_mean",
            "final_acc_per_total_tflop_mean",
            "best_acc_per_tflop_to_best_mean",
        ])

        for label, rs in groups.items():
            def m(attr: str) -> float | None:
                vals = [getattr(r, attr) for r in rs]
                vals = [
                    float(v) for v in vals if v is not None and math.isfinite(float(v))]
                return mean(vals) if vals else None

            def s(attr: str) -> float | None:
                vals = [getattr(r, attr) for r in rs]
                vals = [
                    float(v) for v in vals if v is not None and math.isfinite(float(v))]
                return stdev(vals) if len(vals) > 1 else 0.0 if vals else None

            writer.writerow([
                label,
                len(rs),
                m("best_val_accuracy"),
                s("best_val_accuracy"),
                m("final_val_accuracy"),
                s("final_val_accuracy"),
                m("best_epoch"),
                m("tflops_to_best"),
                m("total_tflops"),
                m("time_to_best"),
                m("total_time"),
                m("best_acc_per_total_tflop"),
                m("final_acc_per_total_tflop"),
                m("best_acc_per_tflop_to_best"),
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_RESULT_DIRS,
        help="Result directories or JSON files to scan.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/agnews_summary.csv"),
        help="Where to write aggregate CSV.",
    )
    parser.add_argument(
        "--no-runs",
        action="store_true",
        help="Do not print per-run table.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    runs = collect(paths)

    if not runs:
        raise SystemExit(f"No AG News result JSONs found under: {paths}")

    groups = group_runs(runs)

    print(
        f"Loaded {len(runs)} AG News runs across {len(groups)} config groups.")
    if not args.no_runs:
        print_detailed_runs(runs)

    print_aggregate_table(groups)
    print_efficiency_table(groups)
    print_interpretation(groups)
    print_latex_table(groups)

    write_csv(groups, args.csv)
    print(f"\nWrote CSV summary to {args.csv}")


if __name__ == "__main__":
    main()
