#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional


@dataclass(frozen=True)
class Run:
    path: Path
    task: str
    seed: int

    method: str
    scheduler: str
    label: str
    group: str

    best_val_accuracy: float
    final_val_accuracy: float
    drop_val_accuracy: float
    best_epoch: int

    total_epochs: int
    total_steps: Optional[int]

    tflops_to_best: Optional[float]
    total_tflops: Optional[float]
    time_to_best: Optional[float]
    total_time: Optional[float]

    label_noise_rate: float
    context_mode: str
    episode_length: Optional[int]
    lr_candidates: tuple[float, ...]
    noise_candidates: tuple[float, ...]


TASK_ORDER = ["sst2", "agnews", "agnews_noise"]

TASK_TITLES = {
    "sst2": "SST-2",
    "agnews": "Clean AG News",
    "agnews_noise": "AG News with 20\\% symmetric label noise",
}

TASK_SHORT = {
    "sst2": "SST-2",
    "agnews": "AG News",
    "agnews_noise": "AG News 20\\%",
}

METHOD_ORDER = [
    "AdamW",
    "AdamW + warmup-linear",
    "AEES",
    "AEES + warmup-linear",
    "AEES-LR",
    "AEES-Noise",
    "AEES-Dual",
    "AEES-LR + warmup-linear",
    "AEES-Noise + warmup-linear",
    "AEES-Dual + warmup-linear",
]

MAIN_METHOD_ORDER = [
    "AdamW",
    "AdamW + warmup-linear",
    "AEES",
    "AEES + warmup-linear",
]

AXIS_METHOD_ORDER = [
    "AdamW + warmup-linear",
    "AEES-LR + warmup-linear",
    "AEES-Noise + warmup-linear",
    "AEES-Dual + warmup-linear",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def cfg(data: dict[str, Any]) -> dict[str, Any]:
    c = data.get("config", {})
    return c if isinstance(c, dict) else {}


def as_float_list(x: Any) -> list[float]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def tuple_float(x: Any) -> tuple[float, ...]:
    if x is None:
        return tuple()
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                pass
        return tuple(out)
    return tuple()


def parse_seed(data: dict[str, Any], path: Path) -> int:
    c = cfg(data)
    if "seed" in data:
        return int(data["seed"])
    if "seed" in c:
        return int(c["seed"])
    m = re.search(r"seed(\d+)", path.name)
    if m:
        return int(m.group(1))
    return -1


def infer_task(data: dict[str, Any], path: Path) -> Optional[str]:
    c = cfg(data)
    task = str(data.get("task_name", c.get("task_name", ""))).lower()
    dataset = str(data.get("dataset_name", c.get("dataset_name", ""))).lower()
    p = str(path).lower()

    if "agnews" in task or "ag news" in dataset or "agnews" in p or "ag_news" in p:
        noise = float(c.get("label_noise_rate", 0.0) or 0.0)
        if noise > 0 or "noisy" in p or "noise" in p:
            return "agnews_noise"
        return "agnews"

    if "sst2" in task or "sst-2" in dataset or "sst2" in p:
        return "sst2"

    return None


def is_close_tuple(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def fmt_arm_tuple(xs: tuple[float, ...]) -> str:
    if not xs:
        return "--"
    return ", ".join(f"{x:g}" for x in xs)


def infer_method_label(data: dict[str, Any], path: Path) -> tuple[str, str, str]:
    """
    Returns:
      method_name, scheduler_name, clean_table_label
    """
    c = cfg(data)

    method = str(data.get("method_name", c.get("method", "unknown")))
    scheduler = str(c.get("lr_scheduler", "none"))
    lr = tuple_float(c.get("lr_candidates"))
    noise = tuple_float(c.get("noise_candidates"))

    sched_suffix = "" if scheduler == "none" else " + warmup-linear" if scheduler == "warmup_linear" else f" + {scheduler}"

    if method == "AdamW":
        return method, scheduler, f"AdamW{sched_suffix}"

    if method == "AdaptiveScheduler":
        lr_only = len(lr) > 1 and is_close_tuple(noise, (0.0,))
        noise_only = is_close_tuple(lr, (1.0,)) and len(noise) > 1
        dual = len(lr) > 1 and len(noise) > 1

        if lr_only:
            return method, scheduler, f"AEES-LR{sched_suffix}"
        if noise_only:
            return method, scheduler, f"AEES-Noise{sched_suffix}"
        if dual:
            return method, scheduler, f"AEES-Dual{sched_suffix}"

        return method, scheduler, f"AEES{sched_suffix}"

    if method == "FixedMode":
        fixed = c.get("fixed_mode_name", None)
        label = "Fixed mode" if fixed is None else f"Fixed mode ({fixed})"
        return method, scheduler, f"{label}{sched_suffix}"

    return method, scheduler, path.stem


def canonical_main_label(label: str) -> str:
    """
    For main tables, collapse AEES-Dual to AEES.
    Axis-specific labels are preserved in axis tables and appendix.
    """
    if label == "AEES-Dual":
        return "AEES"
    if label == "AEES-Dual + warmup-linear":
        return "AEES + warmup-linear"
    return label


def summarize(path: Path) -> Optional[Run]:
    data = load_json(path)
    c = cfg(data)

    task = infer_task(data, path)
    if task is None:
        return None

    val = as_float_list(data.get("val_accuracies"))
    if val:
        best_i = max(range(len(val)), key=lambda i: val[i])
        best_epoch = best_i + 1
        best = float(data.get("best_val_accuracy", val[best_i]))
        final = float(data.get("final_val_accuracy", val[-1]))
    else:
        best = float(data["best_val_accuracy"])
        final = float(data["final_val_accuracy"])
        best_epoch = -1

    epoch_tflops = as_float_list(data.get("epoch_tflops"))
    epoch_times = as_float_list(data.get("epoch_wall_clock_seconds"))

    if data.get("estimated_flops") is not None:
        total_tflops = float(data["estimated_flops"]) / 1e12
    elif epoch_tflops:
        total_tflops = sum(epoch_tflops)
    else:
        total_tflops = None

    tflops_to_best = None
    if epoch_tflops and best_epoch > 0:
        tflops_to_best = sum(epoch_tflops[:best_epoch])

    if data.get("wall_clock_seconds") is not None:
        total_time = float(data["wall_clock_seconds"])
    elif epoch_times:
        total_time = sum(epoch_times)
    else:
        total_time = None

    time_to_best = None
    if epoch_times and best_epoch > 0:
        time_to_best = sum(epoch_times[:best_epoch])

    method, scheduler, label = infer_method_label(data, path)

    return Run(
        path=path,
        task=task,
        seed=parse_seed(data, path),
        method=method,
        scheduler=scheduler,
        label=label,
        group=task,
        best_val_accuracy=best,
        final_val_accuracy=final,
        drop_val_accuracy=best - final,
        best_epoch=best_epoch,
        total_epochs=int(data.get("total_epochs", c.get("epochs", -1))),
        total_steps=int(data["total_steps"]) if data.get(
            "total_steps") is not None else None,
        tflops_to_best=tflops_to_best,
        total_tflops=total_tflops,
        time_to_best=time_to_best,
        total_time=total_time,
        label_noise_rate=float(c.get("label_noise_rate", 0.0) or 0.0),
        context_mode=str(c.get("context_mode", "none")),
        episode_length=int(c["episode_length"]) if c.get(
            "episode_length") is not None else None,
        lr_candidates=tuple_float(c.get("lr_candidates")),
        noise_candidates=tuple_float(c.get("noise_candidates")),
    )


def collect(paths: list[Path]) -> list[Run]:
    runs: list[Run] = []

    for root in paths:
        if root.is_file() and root.suffix == ".json":
            try:
                r = summarize(root)
                if r:
                    runs.append(r)
            except Exception as e:
                print(f"[WARN] failed to parse {root}: {e}")
            continue

        if not root.exists():
            print(f"[WARN] missing path: {root}")
            continue

        for p in sorted(root.rglob("*.json")):
            try:
                r = summarize(p)
                if r:
                    runs.append(r)
            except Exception as e:
                print(f"[WARN] failed to parse {p}: {e}")

    return runs


def mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    return mean(vals), stdev(vals) if len(vals) > 1 else 0.0


def fmt_pm(vals: list[float], *, percent: bool = False, digits: int = 2) -> str:
    m, s = mean_std(vals)
    if m is None or s is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{m * scale:.{digits}f} $\\pm$ {s * scale:.{digits}f}"


def fmt_mean(vals: list[float], *, percent: bool = False, digits: int = 2) -> str:
    m, _ = mean_std(vals)
    if m is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{m * scale:.{digits}f}"


def fmt_scheduler(scheduler: str) -> str:
    if scheduler == "none":
        return "None"
    if scheduler == "warmup_linear":
        return "Warmup-linear"
    return scheduler.replace("_", "-")


def fmt_optional_int(x: Optional[int]) -> str:
    return "--" if x is None else str(x)


def fmt_context(x: str) -> str:
    return "None" if x in {"", "none", "None"} else x


def latex_escape(s: str) -> str:
    if "\\" in s:
        return s
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def method_sort_key(label: str, order: list[str]) -> tuple[int, str]:
    if label in order:
        return order.index(label), label
    return len(order), label


def group_by_label(runs: list[Run], *, collapse_dual: bool = False) -> dict[str, list[Run]]:
    g: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        label = canonical_main_label(r.label) if collapse_dual else r.label
        g[label].append(r)
    return dict(g)


def table_rows(groups: dict[str, list[Run]], order: list[str]) -> list[tuple[str, list[Run]]]:
    return sorted(groups.items(), key=lambda kv: method_sort_key(kv[0], order))


def rank_labels(
    groups: dict[str, list[Run]],
    metric_fn,
    *,
    higher_is_better: bool = True,
) -> dict[str, str]:
    vals: list[tuple[str, float]] = []
    for label, rs in groups.items():
        values = [metric_fn(r) for r in rs]
        m, _ = mean_std(values)
        if m is not None:
            vals.append((label, m))

    vals.sort(key=lambda x: x[1], reverse=higher_is_better)

    styles = {label: "" for label in groups}
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


def main_caption(task: str) -> str:
    if task == "sst2":
        return (
            "SST-2 fine-tuning results using AdamW as the base optimizer. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    if task == "agnews":
        return (
            "Clean AG News fine-tuning results using AdamW as the base optimizer. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    if task == "agnews_noise":
        return (
            "AG News fine-tuning results under 20\\% symmetric training-label noise, with clean validation labels and AdamW as the base optimizer. "
            "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
            "Drop denotes the best-to-final validation decrease in percentage points, where lower is better."
        )
    raise ValueError(f"Unknown task: {task}")


def make_main_table(
    *,
    task: str,
    label: str,
    runs: list[Run],
) -> str:
    groups_all = group_by_label(runs, collapse_dual=True)
    groups = {k: v for k, v in groups_all.items() if k in MAIN_METHOD_ORDER}

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{{main_caption(task)} "
        "Bold and italics mark the best and second-best values in the table, respectively.}}"
    )
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Best (\%) & Final (\%) & Drop (pp) & Peak epoch \\"
    )
    lines.append(r"\midrule")

    best_styles = rank_labels(
        groups, lambda r: r.best_val_accuracy, higher_is_better=True)
    final_styles = rank_labels(
        groups, lambda r: r.final_val_accuracy, higher_is_better=True)
    drop_styles = rank_labels(
        groups, lambda r: r.drop_val_accuracy, higher_is_better=False)

    for method, rs in table_rows(groups, MAIN_METHOD_ORDER):
        best = styled_cell(
            fmt_pm([r.best_val_accuracy for r in rs], percent=True, digits=2),
            best_styles.get(method, ""),
        )
        final = styled_cell(
            fmt_pm([r.final_val_accuracy for r in rs], percent=True, digits=2),
            final_styles.get(method, ""),
        )
        drop = styled_cell(
            fmt_pm([r.drop_val_accuracy for r in rs], percent=True, digits=2),
            drop_styles.get(method, ""),
        )
        best_epoch = fmt_pm([float(r.best_epoch)
                            for r in rs if r.best_epoch > 0], digits=2)

        lines.append(
            f"{latex_escape(method)} & {best} & {final} & {drop} & {best_epoch} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def axis_caption(task: str) -> str:
    if task == "sst2":
        return (
            "Axis-ablation results for SST-2. "
            "All configurations use AdamW with warmup-linear scheduling as the base setup. "
            "\\method{}-LR adapts only the learning-rate multiplier, \\method{}-Noise adapts only gradient-noise magnitude, and \\method{}-Dual adapts both axes."
        )
    if task == "agnews_noise":
        return (
            "Axis-ablation results for AG News under 20\\% symmetric training-label noise. "
            "All configurations use AdamW with warmup-linear scheduling as the base setup. "
            "\\method{}-LR adapts only the learning-rate multiplier, \\method{}-Noise adapts only gradient-noise magnitude, and \\method{}-Dual adapts both axes."
        )
    if task == "agnews":
        return (
            "Axis-ablation results for clean AG News. "
            "All configurations use AdamW with warmup-linear scheduling as the base setup. "
            "\\method{}-LR adapts only the learning-rate multiplier, \\method{}-Noise adapts only gradient-noise magnitude, and \\method{}-Dual adapts both axes."
        )
    raise ValueError(f"Unknown task: {task}")


def make_axis_table(
    *,
    task: str,
    label: str,
    runs: list[Run],
) -> str:
    groups_all = group_by_label(runs, collapse_dual=False)
    groups = {k: v for k, v in groups_all.items() if k in AXIS_METHOD_ORDER}

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        f"\\caption{{{axis_caption(task)} "
        "Best and final validation accuracy are reported as mean $\\pm$ standard deviation across seeds. "
        "Drop denotes the best-to-final validation decrease in percentage points, where lower is better. "
        "Bold and italics mark the best and second-best values in the table, respectively.}}"
    )
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Best (\%) & Final (\%) & Drop (pp) & Peak epoch \\"
    )
    lines.append(r"\midrule")

    best_styles = rank_labels(
        groups, lambda r: r.best_val_accuracy, higher_is_better=True)
    final_styles = rank_labels(
        groups, lambda r: r.final_val_accuracy, higher_is_better=True)
    drop_styles = rank_labels(
        groups, lambda r: r.drop_val_accuracy, higher_is_better=False)

    for method, rs in table_rows(groups, AXIS_METHOD_ORDER):
        best = styled_cell(
            fmt_pm([r.best_val_accuracy for r in rs], percent=True, digits=2),
            best_styles.get(method, ""),
        )
        final = styled_cell(
            fmt_pm([r.final_val_accuracy for r in rs], percent=True, digits=2),
            final_styles.get(method, ""),
        )
        drop = styled_cell(
            fmt_pm([r.drop_val_accuracy for r in rs], percent=True, digits=2),
            drop_styles.get(method, ""),
        )
        best_epoch = fmt_pm([float(r.best_epoch)
                            for r in rs if r.best_epoch > 0], digits=2)

        lines.append(
            f"{latex_escape(method)} & {best} & {final} & {drop} & {best_epoch} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def short_appendix_method_label(method: str) -> str:
    return method.replace(" + warmup-linear", " + WL")


def make_detailed_appendix_table(runs: list[Run]) -> str:
    groups_by_task: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        groups_by_task[r.task].append(r)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Detailed NLP fine-tuning results for SST-2, clean AG News, and AG News with 20\% symmetric training-label noise. "
        r"The table reports the number of seeds $n$, validation metrics, best-to-final validation drop in percentage points, peak epoch, episode length, and the LR/noise candidate axes used by each configuration. "
        r"All configurations use AdamW as the base optimizer. WL denotes warmup-linear learning-rate scheduling.}"
    )
    lines.append(r"\label{tab:nlp_detailed_appendix}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccccccl}")
    lines.append(r"\toprule")
    lines.append(
        r"Task & Method & $n$ & Best & Final & Drop & Peak epoch & Episode length & Axes \\"
    )
    lines.append(
        r"& & & (\%) & (\%) & (pp) & & & \\"
    )
    lines.append(r"\midrule")

    first_task = True

    for task in TASK_ORDER:
        task_runs = groups_by_task.get(task, [])
        if not task_runs:
            continue

        if not first_task:
            lines.append(r"\midrule")
        first_task = False

        groups = group_by_label(task_runs, collapse_dual=False)
        rows = table_rows(groups, METHOD_ORDER)

        for row_idx, (method, rs) in enumerate(rows):
            task_col = TASK_SHORT[task] if row_idx == 0 else ""

            lr_set = sorted({fmt_arm_tuple(r.lr_candidates) for r in rs})
            noise_set = sorted({fmt_arm_tuple(r.noise_candidates) for r in rs})
            axes = f"LR: {'/'.join(lr_set)}; noise: {'/'.join(noise_set)}"

            episode_set = sorted(
                {fmt_optional_int(r.episode_length) for r in rs})
            episode = "/".join(episode_set)

            method_label = short_appendix_method_label(method)

            lines.append(
                f"{task_col} & "
                f"{latex_escape(method_label)} & "
                f"{len(rs)} & "
                f"{fmt_pm([r.best_val_accuracy for r in rs], percent=True, digits=2)} & "
                f"{fmt_pm([r.final_val_accuracy for r in rs], percent=True, digits=2)} & "
                f"{fmt_pm([r.drop_val_accuracy for r in rs], percent=True, digits=2)} & "
                f"{fmt_pm([float(r.best_epoch) for r in rs if r.best_epoch > 0], digits=2)} & "
                f"{latex_escape(episode)} & "
                f"{latex_escape(axes)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def write_csv(path: Path, runs: list[Run]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = group_by_label(runs, collapse_dual=False)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label",
            "n",
            "best_val_accuracy_mean",
            "best_val_accuracy_std",
            "final_val_accuracy_mean",
            "final_val_accuracy_std",
            "drop_best_to_final_mean",
            "drop_best_to_final_std",
            "best_epoch_mean",
            "best_epoch_std",
            "tflops_to_best_mean",
            "tflops_to_best_std",
            "total_tflops_mean",
            "total_tflops_std",
            "time_to_best_mean",
            "time_to_best_std",
            "total_time_mean",
            "total_time_std",
        ])

        for method, rs in table_rows(groups, METHOD_ORDER):
            def ms(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
                return mean_std(vals)

            best_m, best_s = ms([r.best_val_accuracy for r in rs])
            final_m, final_s = ms([r.final_val_accuracy for r in rs])
            drop_m, drop_s = ms([r.drop_val_accuracy for r in rs])
            ep_m, ep_s = ms([float(r.best_epoch)
                            for r in rs if r.best_epoch > 0])
            tb_m, tb_s = ms(
                [r.tflops_to_best for r in rs if r.tflops_to_best is not None])
            tt_m, tt_s = ms(
                [r.total_tflops for r in rs if r.total_tflops is not None])
            timeb_m, timeb_s = ms(
                [r.time_to_best for r in rs if r.time_to_best is not None])
            timet_m, timet_s = ms(
                [r.total_time for r in rs if r.total_time is not None])

            w.writerow([
                method,
                len(rs),
                best_m,
                best_s,
                final_m,
                final_s,
                drop_m,
                drop_s,
                ep_m,
                ep_s,
                tb_m,
                tb_s,
                tt_m,
                tt_s,
                timeb_m,
                timeb_s,
                timet_m,
                timet_s,
            ])


def write_all_tables(out_dir: Path, runs: list[Run]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_task[r.task].append(r)

    tables: dict[str, str] = {}

    if by_task.get("sst2"):
        tables["sst2_main.tex"] = make_main_table(
            task="sst2",
            label="tab:sst2",
            runs=by_task["sst2"],
        )
        tables["sst2_axis.tex"] = make_axis_table(
            task="sst2",
            label="tab:sst2-axis",
            runs=by_task["sst2"],
        )

    if by_task.get("agnews"):
        tables["agnews_main.tex"] = make_main_table(
            task="agnews",
            label="tab:agnews-clean",
            runs=by_task["agnews"],
        )

        axis_groups = group_by_label(by_task["agnews"], collapse_dual=False)
        if any(k in AXIS_METHOD_ORDER for k in axis_groups):
            tables["agnews_axis.tex"] = make_axis_table(
                task="agnews",
                label="tab:agnews-clean-axis",
                runs=by_task["agnews"],
            )

    if by_task.get("agnews_noise"):
        tables["agnews_noise20_main.tex"] = make_main_table(
            task="agnews_noise",
            label="tab:agnews-noise20",
            runs=by_task["agnews_noise"],
        )
        tables["agnews_noise20_axis.tex"] = make_axis_table(
            task="agnews_noise",
            label="tab:agnews-noise20-axis",
            runs=by_task["agnews_noise"],
        )

    tables["nlp_detailed_appendix_table.tex"] = make_detailed_appendix_table(
        runs)

    combined: list[str] = []
    combined.append("% Auto-generated LaTeX tables for NLP experiments.")
    combined.append("% Include with \\input{...} or copy into thesis.")
    combined.append("")

    for name, content in tables.items():
        (out_dir / name).write_text(content + "\n")
        combined.append(f"% ===== {name} =====")
        combined.append(content)
        combined.append("")

    (out_dir / "all_nlp_tables.tex").write_text("\n".join(combined))

    for task, task_runs in by_task.items():
        write_csv(out_dir / f"{task}_summary.csv", task_runs)


def print_summary(runs: list[Run]) -> None:
    by_task: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_task[r.task].append(r)

    for task in TASK_ORDER:
        rs = by_task.get(task, [])
        if not rs:
            continue

        print(f"\n{task}: {len(rs)} runs")
        groups = group_by_label(rs, collapse_dual=False)
        for label, gr in table_rows(groups, METHOD_ORDER):
            best = fmt_pm([r.best_val_accuracy for r in gr], percent=True)
            final = fmt_pm([r.final_val_accuracy for r in gr], percent=True)
            drop = fmt_pm([r.drop_val_accuracy for r in gr], percent=True)
            print(
                f"  {label:35s} n={len(gr):2d} best={best:20s} final={final:20s} drop={drop:20s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "results/clean_agnews",
            "results/noisy_agnews",
            "results/sst2",
        ],
        help="Result folders/files to scan.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tables/nlp_tables"),
        help="Output directory for .tex and .csv tables.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    runs = collect(paths)

    if not runs:
        raise SystemExit("No matching NLP result JSON files found.")

    print_summary(runs)
    write_all_tables(args.out_dir, runs)

    print(f"\nWrote LaTeX and CSV tables to: {args.out_dir}")
    print(f"Combined LaTeX file: {args.out_dir / 'all_nlp_tables.tex'}")


if __name__ == "__main__":
    main()
