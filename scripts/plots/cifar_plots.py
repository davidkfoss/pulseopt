import json
import os
import re
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "pulseopt_plot_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = CACHE_DIR / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = ROOT / "results" / "phase1b_confirm"
PHASE2_DIR = ROOT / "results" / "phase2_baselines"
PHASE2_RERUN_DIR = ROOT / "results" / "phase2b_noisy_final_scheduler_rerun"
OUTDIR = ROOT / "results" / "plots" / "cifar"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
    }
)


PHASE1_RANKING_ORDER = [
    "clean_coarse_lronly_ep200_none",
    "clean_coarse_lronly_ep200_trend",
    "clean_coarse_lronly_ep100_none",
    "clean_coarse_lronly_ep100_trend",
    "clean_main_lronly_ep200_trend",
    "clean_coarse_lronly_ep50_none",
    "clean_main_lronly_ep100_trend",
    "clean_coarse_lronly_ep50_trend",
    "clean_tight_lronly_ep200_trend",
]

PHASE2_ORDER = [
    "adamw_none",
    "adamw_cosine",
    "adamw_linear",
    "aees_none",
    "aees_cosine",
    "aees_linear",
]

PHASE1_SETUP_COLORS = {
    "clean_coarse_lronly_ep200_none": "#1f77b4",
    "clean_coarse_lronly_ep200_trend": "#aec7e8",
    "clean_coarse_lronly_ep100_none": "#ff7f0e",
    "clean_coarse_lronly_ep100_trend": "#ffbb78",
    "clean_coarse_lronly_ep50_none": "#2ca02c",
    "clean_coarse_lronly_ep50_trend": "#98df8a",
    "clean_main_lronly_ep100_trend": "#d62728",
    "clean_main_lronly_ep200_trend": "#ff9896",
    "clean_tight_lronly_ep200_trend": "#9467bd",
}

PHASE2_COLORS = {
    "adamw_none": "#7fb3d8",
    "adamw_cosine": "#2171b5",
    "adamw_linear": "#4292c6",
    "aees_none": "#fc9272",
    "aees_cosine": "#cb181d",
    "aees_linear": "#ef3b2c",
}

PHASE2_LABELS = {
    "adamw_none": "AdamW",
    "adamw_cosine": "AdamW+Cos",
    "adamw_linear": "AdamW+Lin",
    "aees_none": "AEES",
    "aees_cosine": "AEES+Cos",
    "aees_linear": "AEES+Lin",
}

PHASE2_STYLES = {
    "adamw_none": {"color": "#7fb3d8", "ls": "-", "lw": 1.8, "label": "AdamW (no sched)"},
    "adamw_cosine": {"color": "#2171b5", "ls": "-", "lw": 2.2, "label": "AdamW + Cosine"},
    "adamw_linear": {"color": "#4292c6", "ls": "--", "lw": 2.0, "label": "AdamW + Linear"},
    "aees_none": {"color": "#fc9272", "ls": "-", "lw": 1.8, "label": "AEES (no sched)"},
    "aees_cosine": {"color": "#cb181d", "ls": "-", "lw": 2.2, "label": "AEES + Cosine"},
    "aees_linear": {"color": "#ef3b2c", "ls": "--", "lw": 2.0, "label": "AEES + Linear"},
}

PHASE2_OPT_COLORS = {"adamw": "#2171b5", "aees": "#cb181d"}
PHASE2_MARKERS = {
    "adamw_none": "o",
    "adamw_cosine": "s",
    "adamw_linear": "D",
    "aees_none": "o",
    "aees_cosine": "s",
    "aees_linear": "D",
}


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.png")
    fig.savefig(OUTDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def short_label_phase1(setup):
    return setup.replace("clean_", "").replace("_lronly", "")


def normalize_phase2_run(run):
    val_accs = np.array(run["val_accuracies"])
    return {
        "seed": run["seed"],
        "best_val": run["best_val_accuracy"],
        "final_val": run["final_val_accuracy"],
        "final10_val_mean": float(np.mean(val_accs[-10:])),
        "val_accs": val_accs,
        "train_accs": np.array(run["train_accuracies"]),
        "train_losses": np.array(run["train_losses"]),
        "episode_logs": run.get("episode_logs"),
        "controller_logs": run.get("controller_logs"),
    }


def load_phase1_results():
    all_data = {}
    for json_path in sorted(PHASE1_DIR.glob("seed_*/*.json")):
        match = re.search(r"_seed(\d+)\.json$", json_path.name)
        if not match:
            continue
        seed = int(match.group(1))
        setup = re.sub(r"_seed\d+\.json$", "", json_path.name)
        with open(json_path, encoding="utf-8") as f:
            all_data[(setup, seed)] = json.load(f)

    seeds = sorted({seed for _, seed in all_data.keys()})
    missing = []
    for setup in PHASE1_RANKING_ORDER:
        for seed in seeds:
            if (setup, seed) not in all_data:
                missing.append(f"{setup} seed {seed}")
    if missing:
        raise FileNotFoundError(
            "Missing phase1b_confirm CIFAR files:\n" + "\n".join(missing[:20])
        )

    return all_data, seeds


def phase2_source_priority(path):
    path_str = str(path)
    if "phase2b_noisy_final_scheduler_rerun" in path_str:
        return 3
    if "depricated" in path_str:
        return 1
    return 2


def load_phase2_results():
    indexed = {}
    pattern = re.compile(
        r"^(?:cifar_)?(?P<data_type>clean|noisy)_(?P<optimizer>adamw|aees)_(?P<scheduler>none|cosine|linear)(?:_contextnone)?_seed(?P<seed>\d+)\.json$"
    )

    candidate_paths = list(PHASE2_DIR.rglob("*.json")) + list(PHASE2_RERUN_DIR.rglob("*.json"))
    for json_path in sorted(candidate_paths):
        match = pattern.match(json_path.name)
        if not match:
            continue

        data_type = match.group("data_type")
        optimizer = match.group("optimizer")
        scheduler = match.group("scheduler")
        seed = int(match.group("seed"))
        key = f"{data_type}_{optimizer}_{scheduler}"
        priority = phase2_source_priority(json_path)
        prev = indexed.get((key, seed))
        if prev and prev["priority"] > priority:
            continue

        with open(json_path, encoding="utf-8") as f:
            run = json.load(f)
        indexed[(key, seed)] = {
            "priority": priority,
            "path": json_path,
            "run": normalize_phase2_run(run),
        }

    all_data = {key: [] for key in [f"{dt}_{setup}" for dt in ["clean", "noisy"] for setup in PHASE2_ORDER]}
    missing = []
    for key in all_data:
        for seed in range(5):
            item = indexed.get((key, seed))
            if item is None:
                missing.append(f"{key} seed {seed}")
            else:
                all_data[key].append(item["run"])

    if missing:
        raise FileNotFoundError(
            "Missing phase2 CIFAR files:\n" + "\n".join(missing[:30])
        )

    for key in all_data:
        all_data[key].sort(key=lambda run: run["seed"])

    return all_data


def load_phase2_aees_runs(all_data, data_type, scheduler):
    return all_data[f"{data_type}_aees_{scheduler}"]


def main():
    phase1_data, phase1_seeds = load_phase1_results()
    phase2_data = load_phase2_results()

    # PLOT 1: Val accuracy curves (all setups, mean +- std)
    fig, ax = plt.subplots(figsize=(12, 6))
    for setup in PHASE1_RANKING_ORDER:
        curves = []
        for seed in phase1_seeds:
            run = phase1_data.get((setup, seed))
            if run and "val_accuracies" in run:
                curves.append(run["val_accuracies"])
        if not curves:
            continue
        min_len = min(len(curve) for curve in curves)
        curves = np.array([curve[:min_len] for curve in curves])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        epochs = np.arange(1, min_len + 1)

        color = PHASE1_SETUP_COLORS.get(setup, "#333333")
        ax.plot(epochs, mean, color=color, label=short_label_phase1(setup), linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("CIFAR-100 Validation Accuracy - All Setups (mean +- std, 5 seeds)", fontsize=13)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 100)
    ax.set_ylim(0.1, 0.75)
    fig.tight_layout()
    save(fig, "plot1_val_accuracy_all")
    print("Plot 1 done")

    # PLOT 2: Bar chart - best val accuracy per setup (mean +- std)
    fig, ax = plt.subplots(figsize=(12, 5))
    means, stds, labels, colors = [], [], [], []
    for setup in PHASE1_RANKING_ORDER:
        bests = [phase1_data[(setup, seed)]["best_val_accuracy"] for seed in phase1_seeds]
        means.append(np.mean(bests))
        stds.append(np.std(bests))
        labels.append(short_label_phase1(setup))
        colors.append(PHASE1_SETUP_COLORS.get(setup, "#333"))

    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("CIFAR-100 Best Val Accuracy by Setup (mean +- std, 5 seeds)", fontsize=13)
    ax.set_ylim(0.69, 0.73)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.001, f"{mean:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    save(fig, "plot2_best_val_bar")
    print("Plot 2 done")

    # PLOT 3: Arm selection frequencies over time (winning config, all seeds)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    fig.suptitle("Arm Selection Over Time - coarse_ep200_none (each seed)", fontsize=13, y=1.02)
    arm_colors = {"0.5": "#2ca02c", "1.0": "#1f77b4", "2.0": "#d62728"}
    arm_labels = {0.5: "lrx0.5", 1.0: "lrx1.0", 2.0: "lrx2.0"}

    for si, seed in enumerate(phase1_seeds):
        ax = axes[si]
        run = phase1_data[("clean_coarse_lronly_ep200_none", seed)]
        selected = run["episode_logs"]["selected_lr_values"]
        n_episodes = len(selected)
        window = 20
        for arm_val in [0.5, 1.0, 2.0]:
            freqs = []
            for i in range(n_episodes):
                start = max(0, i - window + 1)
                chunk = selected[start : i + 1]
                freqs.append(sum(1 for x_val in chunk if x_val == arm_val) / len(chunk))
            ax.plot(range(n_episodes), freqs, color=arm_colors[str(arm_val)], label=arm_labels[arm_val], linewidth=1.2, alpha=0.85)

        ax.set_title(f"Seed {seed}", fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        if si == 0:
            ax.set_ylabel("Selection Frequency\n(rolling window=20)", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if si == 4:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    save(fig, "plot3_arm_selection_winner")
    print("Plot 3 done")

    # PLOT 4: Reward traces (winning config, all seeds)
    fig, ax = plt.subplots(figsize=(12, 4))
    seed_cmap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for si, seed in enumerate(phase1_seeds):
        run = phase1_data[("clean_coarse_lronly_ep200_none", seed)]
        rewards = run["episode_logs"]["episode_rewards"]
        alpha = 0.1
        smoothed = [rewards[0]]
        for reward in rewards[1:]:
            smoothed.append(alpha * reward + (1 - alpha) * smoothed[-1])

        ax.plot(range(len(rewards)), rewards, color=seed_cmap[si], alpha=0.15, linewidth=0.5)
        ax.plot(range(len(smoothed)), smoothed, color=seed_cmap[si], alpha=0.9, linewidth=1.5, label=f"Seed {seed} (EMA)")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Episode Rewards - coarse_ep200_none (raw + EMA smoothed)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "plot4_reward_traces")
    print("Plot 4 done")

    # PLOT 5: UCB value estimates over time (winning config, seed 0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("UCB Controller Internals - coarse_ep200_none, seed 0", fontsize=13)

    run = phase1_data[("clean_coarse_lronly_ep200_none", 0)]
    controller_logs = run["controller_logs"]["lr_controller_logs"]

    value_estimates = controller_logs["value_estimates_history"]
    for arm_key, arm_label, color in [("0.5", "lrx0.5", "#2ca02c"), ("1.0", "lrx1.0", "#1f77b4"), ("2.0", "lrx2.0", "#d62728")]:
        vals = [episode[arm_key] for episode in value_estimates]
        ax1.plot(range(len(vals)), vals, color=color, label=arm_label, linewidth=1.2)

    ax1.set_ylabel("Value Estimate (Q)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Discounted UCB Value Estimates", fontsize=11)

    effective_counts = controller_logs["effective_counts_history"]
    for arm_key, arm_label, color in [("0.5", "lrx0.5", "#2ca02c"), ("1.0", "lrx1.0", "#1f77b4"), ("2.0", "lrx2.0", "#d62728")]:
        vals = [episode[arm_key] for episode in effective_counts]
        ax2.plot(range(len(vals)), vals, color=color, label=arm_label, linewidth=1.2)

    ax2.set_xlabel("Controller Update Step", fontsize=11)
    ax2.set_ylabel("Effective Count (discounted)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Discounted Effective Counts per Arm", fontsize=11)

    fig.tight_layout()
    save(fig, "plot5_ucb_internals")
    print("Plot 5 done")

    # PLOT 6: Coarse vs Main vs Tight (head-to-head at ep200)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    compare = [
        ("clean_coarse_lronly_ep200_none", "Coarse [0.5,1,2] ep200 none"),
        ("clean_main_lronly_ep200_trend", "Main [0.7,1,1.3] ep200 trend"),
        ("clean_tight_lronly_ep200_trend", "Tight [0.85,1,1.2] ep200 trend"),
    ]
    compare_colors = ["#1f77b4", "#ff9896", "#9467bd"]

    ax = axes[0]
    for (setup, label), color in zip(compare, compare_colors):
        curves = [phase1_data[(setup, seed)]["val_accuracies"] for seed in phase1_seeds]
        min_len = min(len(curve) for curve in curves)
        curves = np.array([curve[:min_len] for curve in curves])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        epochs = np.arange(1, min_len + 1)
        ax.plot(epochs, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("LR Range Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.1, 0.75)

    ax = axes[1]
    for (setup, label), color in zip(compare, compare_colors):
        run = phase1_data[(setup, 0)]
        selected = run["episode_logs"]["selected_lr_values"]
        lr_cands = run["config"]["lr_candidates"]
        lowest = min(lr_cands)
        window = 15
        freqs = []
        for i in range(len(selected)):
            start = max(0, i - window + 1)
            chunk = selected[start : i + 1]
            freqs.append(sum(1 for x_val in chunk if x_val == lowest) / len(chunk))
        ax.plot(range(len(freqs)), freqs, color=color, label=f"{label}\n(lowest={lowest}x)", linewidth=1.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Freq. of Lowest LR Arm")
    ax.set_title("Convergence to Conservative Arm (seed 0)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for (setup, label), color in zip(compare, compare_colors):
        curves = [phase1_data[(setup, seed)]["train_losses"] for seed in phase1_seeds]
        min_len = min(len(curve) for curve in curves)
        curves = np.array([curve[:min_len] for curve in curves])
        mean = curves.mean(axis=0)
        epochs = np.arange(1, min_len + 1)
        ax.plot(epochs, mean, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("Coarse vs Main vs Tight - LR Range Comparison (ep200)", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "plot6_lr_range_comparison")
    print("Plot 6 done")

    # FIG 1: Best vs Final Val Bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        x = np.arange(len(PHASE2_ORDER))
        width = 0.35
        best_means, best_stds, final_means, final_stds = [], [], [], []
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            best_vals = [run["best_val"] for run in runs]
            final_vals = [run["final_val"] for run in runs]
            best_means.append(np.mean(best_vals) * 100)
            best_stds.append(np.std(best_vals) * 100)
            final_means.append(np.mean(final_vals) * 100)
            final_stds.append(np.std(final_vals) * 100)
        ax.bar(x - width / 2, best_means, width, yerr=best_stds, color=[PHASE2_COLORS[setup] for setup in PHASE2_ORDER], edgecolor="black", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.2}, alpha=0.95)
        ax.bar(x + width / 2, final_means, width, yerr=final_stds, color=[PHASE2_COLORS[setup] for setup in PHASE2_ORDER], edgecolor="black", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.2}, alpha=0.55, hatch="///")
        for i, best_mean in enumerate(best_means):
            ax.text(i - width / 2, best_mean + 0.3, f"{best_mean:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (20% Label Noise)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ymin = min(best_means + final_means) - 3
        ymax = max(best_means + final_means) + 2.5
        ax.set_ylim(ymin, ymax)
    legend_elements = [
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="Best Val Acc"),
        Patch(facecolor="gray", alpha=0.55, edgecolor="black", hatch="///", label="Final Val Acc"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    save(fig, "fig1_best_vs_final_bars")
    print("Fig 1 done")

    # FIG 1B: Best vs Final 10-Epoch Mean Val Bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        x = np.arange(len(PHASE2_ORDER))
        width = 0.35
        best_means, best_stds, final10_means, final10_stds = [], [], [], []
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            best_vals = [run["best_val"] for run in runs]
            final10_vals = [run["final10_val_mean"] for run in runs]
            best_means.append(np.mean(best_vals) * 100)
            best_stds.append(np.std(best_vals) * 100)
            final10_means.append(np.mean(final10_vals) * 100)
            final10_stds.append(np.std(final10_vals) * 100)
        ax.bar(x - width / 2, best_means, width, yerr=best_stds, color=[PHASE2_COLORS[setup] for setup in PHASE2_ORDER], edgecolor="black", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.2}, alpha=0.95)
        ax.bar(x + width / 2, final10_means, width, yerr=final10_stds, color=[PHASE2_COLORS[setup] for setup in PHASE2_ORDER], edgecolor="black", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.2}, alpha=0.55, hatch="///")
        for i, best_mean in enumerate(best_means):
            ax.text(i - width / 2, best_mean + 0.3, f"{best_mean:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (20% Label Noise)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ymin = min(best_means + final10_means) - 3
        ymax = max(best_means + final10_means) + 2.5
        ax.set_ylim(ymin, ymax)
    legend_elements = [
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="Best Val Acc"),
        Patch(facecolor="gray", alpha=0.55, edgecolor="black", hatch="///", label="Final 10-Epoch Mean Val Acc"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    save(fig, "fig1_best_vs_final10mean_bars")
    print("Fig 1B done")

    # FIG 2: Val Accuracy Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        epochs = np.arange(1, 201)
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            stacked = np.stack([run["val_accs"][:200] for run in runs])
            mean = np.mean(stacked, axis=0) * 100
            std = np.std(stacked, axis=0) * 100
            style = PHASE2_STYLES[setup]
            ax.plot(epochs, mean, color=style["color"], ls=style["ls"], lw=style["lw"], label=style["label"])
            ax.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.1)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (20% Label Noise)"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(1, 200)
    axes[0].legend(fontsize=8.5, loc="lower right", framealpha=0.9, ncol=2)
    plt.tight_layout()
    save(fig, "fig2_val_curves")
    print("Fig 2 done")

    # FIG 3: Generalization Gap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        epochs = np.arange(1, 201)
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            gaps = [run["train_accs"][:200] - run["val_accs"][:200] for run in runs]
            stacked = np.stack(gaps)
            mean = np.mean(stacked, axis=0) * 100
            std = np.std(stacked, axis=0) * 100
            style = PHASE2_STYLES[setup]
            ax.plot(epochs, mean, color=style["color"], ls=style["ls"], lw=style["lw"], label=style["label"])
            ax.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.08)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (20% Label Noise)"
        ax.set_title(f"Generalization Gap - {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Train Acc - Val Acc (%)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(1, 200)
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.9, ncol=2)
    plt.tight_layout()
    save(fig, "fig3_generalization_gap")
    print("Fig 3 done")

    # FIG 4: Heatmap
    schedulers = ["none", "cosine", "linear"]
    optimizers = ["adamw", "aees"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        matrix = np.zeros((2, 3))
        annot = [["" for _ in range(3)] for _ in range(2)]
        for i, optimizer in enumerate(optimizers):
            for j, scheduler in enumerate(schedulers):
                key = f"{data_type}_{optimizer}_{scheduler}"
                runs = phase2_data[key]
                best_vals = [run["best_val"] for run in runs]
                mean = np.mean(best_vals) * 100
                std = np.std(best_vals) * 100
                matrix[i, j] = mean
                annot[i][j] = f"{mean:.2f}\n+-{std:.2f}"
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=matrix.min() - 0.5, vmax=matrix.max() + 0.5)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["No Scheduler", "Cosine", "Linear"], fontsize=10)
        ax.set_yticks(range(2))
        ax.set_yticklabels(["AdamW", "AEES"], fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy)"
        ax.set_title(f"Best Val Accuracy (%) - {title}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save(fig, "fig4_heatmap")
    print("Fig 4 done")

    # FIG 5: AEES Delta
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35
    colors_dt = {"clean": "#4292c6", "noisy": "#ef3b2c"}
    for dt_idx, data_type in enumerate(["clean", "noisy"]):
        deltas, delta_errs = [], []
        for scheduler in ["none", "cosine", "linear"]:
            aees_sorted = sorted(phase2_data[f"{data_type}_aees_{scheduler}"], key=lambda run: run["seed"])
            adamw_sorted = sorted(phase2_data[f"{data_type}_adamw_{scheduler}"], key=lambda run: run["seed"])
            paired = np.array([a_run["best_val"] - b_run["best_val"] for a_run, b_run in zip(aees_sorted, adamw_sorted)])
            deltas.append(np.mean(paired) * 100)
            delta_errs.append(np.std(paired) * 100 / np.sqrt(len(paired)))
        offset = -width / 2 + dt_idx * width
        bars = ax.bar(x + offset, deltas, width, yerr=delta_errs, color=colors_dt[data_type], edgecolor="black", linewidth=0.5, capsize=4, error_kw={"linewidth": 1.5}, alpha=0.85, label=f'{"Clean" if data_type == "clean" else "Noisy (20%)"}')
        for bar, delta in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{delta:+.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["No Scheduler", "Cosine Annealing", "Linear Decay"], fontsize=11)
    ax.set_ylabel("AEES - AdamW (pp)", fontsize=12)
    ax.set_title("Improvement from AEES Wrapper over AdamW\n(Best Val Accuracy, paired by seed +- SEM)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save(fig, "fig5_aees_delta")
    print("Fig 5 done")

    # FIG 6: Best vs Final Scatter
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        ax.plot([50, 80], [50, 80], "k--", alpha=0.3, linewidth=1, label="Best = Final")
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            optimizer = setup.split("_")[0]
            bests = [run["best_val"] * 100 for run in runs]
            finals = [run["final_val"] * 100 for run in runs]
            ax.errorbar(np.mean(bests), np.mean(finals), xerr=np.std(bests), yerr=np.std(finals), marker=PHASE2_MARKERS[setup], color=PHASE2_OPT_COLORS[optimizer], markersize=9, capsize=3, linewidth=1.2, label=PHASE2_LABELS[setup], markeredgecolor="black", markeredgewidth=0.5)
        title = "Clean" if data_type == "clean" else "Noisy (20%)"
        ax.set_title(f"Best vs Final Val Accuracy - {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Best Val Accuracy (%)", fontsize=11)
        ax.set_ylabel("Final Val Accuracy (%)", fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
        all_vals = []
        for setup in PHASE2_ORDER:
            for run in phase2_data[f"{data_type}_{setup}"]:
                all_vals.extend([run["best_val"] * 100, run["final_val"] * 100])
        lo, hi = min(all_vals) - 2, max(all_vals) + 2
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    plt.tight_layout()
    save(fig, "fig6_best_vs_final_scatter")
    print("Fig 6 done")

    # FIG 6B: Best vs Final 10-Epoch Mean Scatter
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        ax.plot([50, 80], [50, 80], "k--", alpha=0.3, linewidth=1, label="Best = Final 10-Epoch Mean")
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            optimizer = setup.split("_")[0]
            bests = [run["best_val"] * 100 for run in runs]
            final10_vals = [run["final10_val_mean"] * 100 for run in runs]
            ax.errorbar(np.mean(bests), np.mean(final10_vals), xerr=np.std(bests), yerr=np.std(final10_vals), marker=PHASE2_MARKERS[setup], color=PHASE2_OPT_COLORS[optimizer], markersize=9, capsize=3, linewidth=1.2, label=PHASE2_LABELS[setup], markeredgecolor="black", markeredgewidth=0.5)
        title = "Clean" if data_type == "clean" else "Noisy (20%)"
        ax.set_title(f"Best vs Final 10-Epoch Mean Val Accuracy - {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Best Val Accuracy (%)", fontsize=11)
        ax.set_ylabel("Final 10-Epoch Mean Val Accuracy (%)", fontsize=11)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
        all_vals = []
        for setup in PHASE2_ORDER:
            for run in phase2_data[f"{data_type}_{setup}"]:
                all_vals.extend([run["best_val"] * 100, run["final10_val_mean"] * 100])
        lo, hi = min(all_vals) - 2, max(all_vals) + 2
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    plt.tight_layout()
    save(fig, "fig6_best_vs_final10mean_scatter")
    print("Fig 6B done")

    # FIG 7: Noisy Zoom + Train Loss
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    epochs = np.arange(1, 51)
    for setup in PHASE2_ORDER:
        key = f"noisy_{setup}"
        runs = phase2_data[key]
        stacked = np.stack([run["val_accs"][:50] for run in runs])
        mean = np.mean(stacked, axis=0) * 100
        std = np.std(stacked, axis=0) * 100
        style = PHASE2_STYLES[setup]
        ax.plot(epochs, mean, color=style["color"], ls=style["ls"], lw=style["lw"], label=style["label"])
        ax.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.12)
        peak_idx = np.argmax(mean)
        ax.plot(epochs[peak_idx], mean[peak_idx], marker="*", color=style["color"], markersize=12, markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.set_title("Noisy CIFAR-100 - Early Training (Epochs 1-50)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.axvline(x=20, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)
    ax.text(21, ax.get_ylim()[0] + 1, "Typical peak\nregion", fontsize=8, color="gray", va="bottom")

    ax = axes[1]
    epochs = np.arange(1, 201)
    for setup in PHASE2_ORDER:
        key = f"noisy_{setup}"
        runs = phase2_data[key]
        stacked = np.stack([run["train_losses"][:200] for run in runs])
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        style = PHASE2_STYLES[setup]
        ax.plot(epochs, mean, color=style["color"], ls=style["ls"], lw=style["lw"], label=style["label"])
        ax.fill_between(epochs, mean - std, mean + std, color=style["color"], alpha=0.08)
    ax.set_title("Noisy CIFAR-100 - Train Loss (Noise Memorization)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    plt.tight_layout()
    save(fig, "fig7_noisy_zoom_and_trainloss")
    print("Fig 7 done")

    # FIG 8: Stability
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    window = 10
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            all_rolling = []
            for run in runs:
                val_accs = run["val_accs"][:200]
                rolling_std = [np.std(val_accs[i - window : i]) for i in range(window, len(val_accs))]
                all_rolling.append(np.array(rolling_std))
            stacked = np.stack(all_rolling)
            mean_rolling = np.mean(stacked, axis=0) * 100
            epochs = np.arange(window + 1, 201)
            style = PHASE2_STYLES[setup]
            ax.plot(epochs, mean_rolling, color=style["color"], ls=style["ls"], lw=style["lw"], label=style["label"])
        title = "Clean" if data_type == "clean" else "Noisy (20%)"
        ax.set_title(f"Validation Accuracy Stability - {title}\n(Rolling {window}-epoch Std Dev)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Rolling Std Dev (%)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(window + 1, 200)
    axes[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9, ncol=2)
    plt.tight_layout()
    save(fig, "fig8_stability")
    print("Fig 8 done")

    # FIG 9: Seed Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels9 = {
        "adamw_none": "AdamW",
        "adamw_cosine": "AdamW\n+Cos",
        "adamw_linear": "AdamW\n+Lin",
        "aees_none": "AEES",
        "aees_cosine": "AEES\n+Cos",
        "aees_linear": "AEES\n+Lin",
    }
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for i, setup in enumerate(PHASE2_ORDER):
            key = f"{data_type}_{setup}"
            runs = phase2_data[key]
            vals = np.array([run["best_val"] * 100 for run in runs])
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals, color=PHASE2_COLORS[setup], s=50, alpha=0.7, edgecolors="black", linewidths=0.5, zorder=3)
            mean = np.mean(vals)
            ax.plot([i - 0.25, i + 0.25], [mean, mean], color=PHASE2_COLORS[setup], linewidth=2.5, zorder=4)
            ax.plot([i - 0.25, i + 0.25], [mean, mean], color="black", linewidth=0.8, zorder=4)
        ax.set_xticks(range(len(PHASE2_ORDER)))
        ax.set_xticklabels([labels9[setup] for setup in PHASE2_ORDER], fontsize=9)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Best Val Accuracy by Seed - {title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.axvline(x=2.5, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)
    plt.tight_layout()
    save(fig, "fig9_seed_distribution")
    print("Fig 9 done")

    # FIG 10: Significance Heatmap
    def sig_label(p_val):
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        n = len(PHASE2_ORDER)
        pval_matrix = np.full((n, n), np.nan)
        diff_matrix = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a_vals = [run["best_val"] for run in phase2_data[f"{data_type}_{PHASE2_ORDER[i]}"]]
                b_vals = [run["best_val"] for run in phase2_data[f"{data_type}_{PHASE2_ORDER[j]}"]]
                _, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
                pval_matrix[i, j] = p_val
                diff_matrix[i, j] = np.mean(a_vals) - np.mean(b_vals)

        im = ax.imshow(pval_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=11, fontweight="bold")
                else:
                    diff = diff_matrix[i, j]
                    p_val = pval_matrix[i, j]
                    sig = sig_label(p_val)
                    color = "white" if p_val < 0.05 else "black"
                    ax.text(j, i, f"{diff:+.4f}\n({sig})", ha="center", va="center", fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(n))
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], fontsize=9)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Pairwise Differences in Best Val Acc\n{title} (row - column, Welch's t-test)", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 0.93, 1])
    fig.colorbar(im, ax=axes, location="right", fraction=0.03, pad=0.02, label="p-value")
    save(fig, "fig10_significance_heatmap")
    print("Fig 10 done")

    # FIG 11: Paired Seed Comparison
    seed_colors = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3", 4: "#ff7f00"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for seed in range(5):
            vals = []
            for setup in PHASE2_ORDER:
                key = f"{data_type}_{setup}"
                seed_run = [run for run in phase2_data[key] if run["seed"] == seed]
                vals.append(seed_run[0]["best_val"] * 100 if seed_run else np.nan)
            ax.plot(range(len(PHASE2_ORDER)), vals, "--", color=seed_colors[seed], alpha=0.6, linewidth=1.2)
            ax.scatter(range(len(PHASE2_ORDER)), vals, color=seed_colors[seed], s=60, zorder=5, edgecolors="black", linewidths=0.5, label=f"Seed {seed}")

        ax.set_xticks(range(len(PHASE2_ORDER)))
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Per-Seed Paired Comparison - {title}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.axvline(x=2.5, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)

    axes[0].legend(fontsize=8.5, loc="lower right", ncol=5)
    plt.tight_layout()
    save(fig, "fig11_paired_seed_comparison")
    print("Fig 11 done")

    # FIG 12: Interaction Plot (Optimizer x Scheduler)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sched_labels = ["No Scheduler", "Cosine", "Linear"]
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for optimizer, color, marker, label in [("adamw", "#2171b5", "s", "AdamW"), ("aees", "#cb181d", "o", "AEES (ours)")]:
            means, stds = [], []
            for scheduler in schedulers:
                key = f"{data_type}_{optimizer}_{scheduler}"
                vals = [run["best_val"] * 100 for run in phase2_data[key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(range(3), means, yerr=stds, marker=marker, color=color, markersize=10, linewidth=2.5, capsize=5, capthick=1.5, label=label, markeredgecolor="black", markeredgewidth=0.5)

        for j, scheduler in enumerate(schedulers):
            aees_mean = np.mean([run["best_val"] * 100 for run in phase2_data[f"{data_type}_aees_{scheduler}"]])
            adamw_mean = np.mean([run["best_val"] * 100 for run in phase2_data[f"{data_type}_adamw_{scheduler}"]])
            delta = aees_mean - adamw_mean
            mid = (aees_mean + adamw_mean) / 2
            ax.annotate(f"Delta = {delta:+.2f}", xy=(j + 0.12, mid), fontsize=9, color="#666666", fontstyle="italic")

        ax.set_xticks(range(3))
        ax.set_xticklabels(sched_labels, fontsize=10)
        ax.set_ylabel("Mean Best Val Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Interaction: Optimizer x LR Scheduler\n{title}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    save(fig, "fig12_interaction_plot")
    print("Fig 12 done")

    # FIG 13: Box + Strip Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        box_data = []
        for setup in PHASE2_ORDER:
            key = f"{data_type}_{setup}"
            box_data.append([run["best_val"] * 100 for run in phase2_data[key]])

        bp = ax.boxplot(box_data, positions=range(len(PHASE2_ORDER)), widths=0.5, patch_artist=True, showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})
        for patch, setup in zip(bp["boxes"], PHASE2_ORDER):
            patch.set_facecolor(PHASE2_COLORS[setup])
            patch.set_alpha(0.6)

        for i, setup in enumerate(PHASE2_ORDER):
            key = f"{data_type}_{setup}"
            vals = [run["best_val"] * 100 for run in phase2_data[key]]
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, color=PHASE2_COLORS[setup], s=45, alpha=0.8, edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_xticks(range(len(PHASE2_ORDER)))
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Best Val Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Distribution of Best Val Accuracy - {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.axvline(x=2.5, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)

    plt.tight_layout()
    save(fig, "fig13_box_strip")
    print("Fig 13 done")

    # FIG 14: Best vs Final - Per-Seed Arrow Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for i, setup in enumerate(PHASE2_ORDER):
            key = f"{data_type}_{setup}"
            runs = sorted(phase2_data[key], key=lambda run: run["seed"])
            for j, run in enumerate(runs):
                best_val = run["best_val"] * 100
                final_val = run["final_val"] * 100
                x_jitter = i + (j - 2) * 0.08
                ax.scatter(x_jitter, best_val, marker="^", color=PHASE2_COLORS[setup], s=40, edgecolors="black", linewidths=0.4, zorder=5)
                ax.scatter(x_jitter, final_val, marker="v", color=PHASE2_COLORS[setup], s=40, edgecolors="black", linewidths=0.4, zorder=5, alpha=0.6)
                ax.plot([x_jitter, x_jitter], [best_val, final_val], color=PHASE2_COLORS[setup], linewidth=1.2, alpha=0.5)

        ax.set_xticks(range(len(PHASE2_ORDER)))
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Best vs Final Val Accuracy per Seed - {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.axvline(x=2.5, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)
        ax.scatter([], [], marker="^", color="gray", s=40, edgecolors="black", linewidths=0.4, label="Best val")
        ax.scatter([], [], marker="v", color="gray", s=40, edgecolors="black", linewidths=0.4, alpha=0.6, label="Final val")
        ax.legend(fontsize=9, loc="lower right" if data_type == "clean" else "upper right")

    plt.tight_layout()
    save(fig, "fig14_best_final_arrows")
    print("Fig 14 done")

    # FIG 14B: Best vs Final 10-Epoch Mean - Per-Seed Arrow Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax_idx, data_type in enumerate(["clean", "noisy"]):
        ax = axes[ax_idx]
        for i, setup in enumerate(PHASE2_ORDER):
            key = f"{data_type}_{setup}"
            runs = sorted(phase2_data[key], key=lambda run: run["seed"])
            for j, run in enumerate(runs):
                best_val = run["best_val"] * 100
                final10_val = run["final10_val_mean"] * 100
                x_jitter = i + (j - 2) * 0.08
                ax.scatter(x_jitter, best_val, marker="^", color=PHASE2_COLORS[setup], s=40, edgecolors="black", linewidths=0.4, zorder=5)
                ax.scatter(x_jitter, final10_val, marker="v", color=PHASE2_COLORS[setup], s=40, edgecolors="black", linewidths=0.4, zorder=5, alpha=0.6)
                ax.plot([x_jitter, x_jitter], [best_val, final10_val], color=PHASE2_COLORS[setup], linewidth=1.2, alpha=0.5)

        ax.set_xticks(range(len(PHASE2_ORDER)))
        ax.set_xticklabels([PHASE2_LABELS[setup] for setup in PHASE2_ORDER], fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
        title = "CIFAR-100 (Clean)" if data_type == "clean" else "CIFAR-100 (Noisy 20%)"
        ax.set_title(f"Best vs Final 10-Epoch Mean Val Accuracy per Seed - {title}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.axvline(x=2.5, color="gray", alpha=0.4, linestyle=":", linewidth=1.5)
        ax.scatter([], [], marker="^", color="gray", s=40, edgecolors="black", linewidths=0.4, label="Best val")
        ax.scatter([], [], marker="v", color="gray", s=40, edgecolors="black", linewidths=0.4, alpha=0.6, label="Final 10-epoch mean val")
        ax.legend(fontsize=9, loc="lower right" if data_type == "clean" else "upper right")

    plt.tight_layout()
    save(fig, "fig14_best_final10mean_arrows")
    print("Fig 14B done")

    # FIG 15: LR Multiplier Selection Probability (corrected y-axis 0-1.0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    bandit_configs = [
        ("clean", "none", "Clean - No Scheduler"),
        ("clean", "cosine", "Clean - Cosine"),
        ("clean", "linear", "Clean - Linear"),
        ("noisy", "none", "Noisy - No Scheduler"),
        ("noisy", "cosine", "Noisy - Cosine"),
        ("noisy", "linear", "Noisy - Linear"),
    ]
    bandit_arm_colors = {"0.5": "#377eb8", "1.0": "#999999", "2.0": "#e41a1c"}
    bandit_arm_labels = {"0.5": "LRx0.5", "1.0": "LRx1.0", "2.0": "LRx2.0"}

    for ax_idx, (data_type, scheduler, title) in enumerate(bandit_configs):
        ax = axes[ax_idx // 3][ax_idx % 3]
        runs = load_phase2_aees_runs(phase2_data, data_type, scheduler)
        if not runs:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            continue

        arm_vals = [str(val) for val in runs[0]["controller_logs"]["lr_controller_logs"]["arm_values"]]
        window = 20
        for arm_val in arm_vals:
            all_freqs = []
            for run in runs:
                selected = np.array(run["episode_logs"]["selected_lr_values"])
                freqs = []
                for i in range(window, len(selected) + 1):
                    freqs.append(np.mean(selected[i - window : i] == float(arm_val)))
                all_freqs.append(np.array(freqs))

            min_len = min(len(freq) for freq in all_freqs)
            stacked = np.stack([freq[:min_len] for freq in all_freqs])
            mean_freq = np.mean(stacked, axis=0)
            std_freq = np.std(stacked, axis=0)
            episodes = np.arange(window, window + min_len)
            ax.plot(episodes, mean_freq, color=bandit_arm_colors[arm_val], linewidth=1.8, label=bandit_arm_labels[arm_val])
            ax.fill_between(episodes, mean_freq - std_freq, mean_freq + std_freq, color=bandit_arm_colors[arm_val], alpha=0.12)

        ax.axhline(y=1 / 3, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Selection Frequency", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0][0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("LR Multiplier Selection Probability Over Training (rolling 20-episode window, mean +- std across seeds)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig15_lr_selection")
    print("Fig 15 done")

    # FIG 16: Bandit Reward Trajectory
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax_idx, (data_type, scheduler, title) in enumerate(bandit_configs):
        ax = axes[ax_idx // 3][ax_idx % 3]
        runs = load_phase2_aees_runs(phase2_data, data_type, scheduler)
        if not runs:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            continue

        for run in runs:
            rewards = np.array(run["episode_logs"]["episode_rewards"])
            color = "#e41a1c" if data_type == "clean" else "#4daf4a"
            ax.plot(range(len(rewards)), rewards, color=color, alpha=0.15, linewidth=0.6)

        min_len = min(len(run["episode_logs"]["episode_rewards"]) for run in runs)
        stacked = np.stack([np.array(run["episode_logs"]["episode_rewards"][:min_len]) for run in runs])
        mean_reward = np.mean(stacked, axis=0)
        smooth_window = 15
        smoothed = np.convolve(mean_reward, np.ones(smooth_window) / smooth_window, mode="valid")
        episodes = np.arange(smooth_window - 1, min_len)[: len(smoothed)]
        ax.plot(episodes, smoothed, color="black", linewidth=2.5, label="Mean (smoothed)")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Episode Reward", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0][0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Bandit Reward Trajectory Over Training (individual seeds + smoothed mean)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig16_reward_trajectory")
    print("Fig 16 done")

    # FIG 17: UCB Value Estimates Evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax_idx, (data_type, scheduler, title) in enumerate(bandit_configs):
        ax = axes[ax_idx // 3][ax_idx % 3]
        runs = load_phase2_aees_runs(phase2_data, data_type, scheduler)
        if not runs:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            continue

        arm_vals = [str(val) for val in runs[0]["controller_logs"]["lr_controller_logs"]["arm_values"]]
        for arm_val in arm_vals:
            all_q = []
            for run in runs:
                q_history = run["controller_logs"]["lr_controller_logs"]["value_estimates_history"]
                q_vals = [step[arm_val] for step in q_history]
                all_q.append(np.array(q_vals))

            min_len = min(len(q_vals) for q_vals in all_q)
            stacked = np.stack([q_vals[:min_len] for q_vals in all_q])
            mean_q = np.mean(stacked, axis=0)
            std_q = np.std(stacked, axis=0)
            episodes = np.arange(min_len)
            ax.plot(episodes, mean_q, color=bandit_arm_colors[arm_val], linewidth=1.8, label=bandit_arm_labels[arm_val])
            ax.fill_between(episodes, mean_q - std_q, mean_q + std_q, color=bandit_arm_colors[arm_val], alpha=0.12)

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Q-value Estimate", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    axes[0][0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("UCB Value Estimates (Q-values) Over Training (mean +- std across seeds)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig17_ucb_values")
    print("Fig 17 done")

    print(f"\nAll CIFAR plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
