from collections import Counter
import glob
import json
import os
from pathlib import Path
import tempfile


CACHE_DIR = Path(tempfile.gettempdir()) / "pulseopt_plot_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = CACHE_DIR / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats


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


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "sst2_last_phase"
PHASE1B_FOLLOWUP_DIR = ROOT / "results" / "sst2_phase1b_followup"
OUTDIR = ROOT / "results" / "plots" / "sst2"

GROUP_ORDER = [
    "AdamW",
    "AdamW + warmup linear",
    "Adaptive (ours)",
    "Adaptive (ours) + warmup linear",
]
SHORT_NAMES = {
    "AdamW": "AdamW",
    "AdamW + warmup linear": "AdamW + WL",
    "Adaptive (ours)": "Adaptive",
    "Adaptive (ours) + warmup linear": "Adaptive + WL",
}
COLORS = {
    "AdamW": "#7f8c8d",
    "AdamW + warmup linear": "#3498db",
    "Adaptive (ours)": "#e74c3c",
    "Adaptive (ours) + warmup linear": "#2ecc71",
}
HATCHES = {
    "AdamW": "//",
    "AdamW + warmup linear": "",
    "Adaptive (ours)": "\\\\",
    "Adaptive (ours) + warmup linear": "",
}


def load_groups():
    files = sorted(glob.glob(str(DATA_DIR / "*.json")))
    if not files:
        raise FileNotFoundError(f"No SST-2 JSON files found in {DATA_DIR}")

    groups = {group_name: [] for group_name in GROUP_ORDER}

    for file_path in files:
        with open(file_path, encoding="utf-8") as fh:
            run = json.load(fh)

        filename = Path(file_path).name
        if "adamw_none" in filename:
            group_name = "AdamW"
        elif "adamw_warmup_linear" in filename:
            group_name = "AdamW + warmup linear"
        elif "context_trend" in filename:
            group_name = "Adaptive (ours)"
        elif "trend_warmup_linear" in filename:
            group_name = "Adaptive (ours) + warmup linear"
        else:
            continue

        groups[group_name].append(run)

    missing_groups = [group_name for group_name in GROUP_ORDER if not groups[group_name]]
    if missing_groups:
        raise ValueError(
            "Missing SST-2 results for: " + ", ".join(missing_groups)
        )

    for group_name in GROUP_ORDER:
        groups[group_name].sort(key=lambda run: run["seed"])

    return groups


def load_phase1b_followup_groups():
    files = sorted(PHASE1B_FOLLOWUP_DIR.glob("seed_*/*.json"))
    if not files:
        raise FileNotFoundError(f"No SST-2 phase1b follow-up JSON files found in {PHASE1B_FOLLOWUP_DIR}")

    groups = {}
    for file_path in files:
        filename = file_path.name
        if "adamw" in filename:
            continue

        with open(file_path, encoding="utf-8") as fh:
            run = json.load(fh)

        group_name = filename.rsplit("_seed", 1)[0]
        groups.setdefault(group_name, []).append(run)

    if not groups:
        raise ValueError("No non-AdamW SST-2 phase1b follow-up runs found")

    for group_name in groups:
        groups[group_name].sort(key=lambda run: run["seed"])

    return groups


def short_name_phase1b_followup(group_name):
    return group_name.replace("sst2_", "")


def add_bracket(ax, x1, x2, y, height, text):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.0, color="black")
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", fontsize=9)


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.png")
    fig.savefig(OUTDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def main():
    groups = load_groups()
    phase1b_followup_groups = load_phase1b_followup_groups()

    def by_seed(group_name):
        return {run["seed"]: run for run in groups[group_name]}

    # PLOT 1: Main result - bar chart with 95% CI
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(GROUP_ORDER))
    means, cis, colors = [], [], []
    for group_name in GROUP_ORDER:
        vals = [run["best_val_accuracy"] for run in groups[group_name]]
        mean = np.mean(vals)
        ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))
        means.append(mean)
        cis.append(ci)
        colors.append(COLORS[group_name])

    bars = ax.bar(
        x_pos,
        means,
        yerr=cis,
        width=0.6,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        error_kw={"linewidth": 1.5},
        zorder=3,
    )
    for i, group_name in enumerate(GROUP_ORDER):
        bars[i].set_hatch(HATCHES[group_name])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER], fontsize=11)
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("SST-2 Classification - Best Validation Accuracy (5 seeds)")
    ax.set_ylim(0.88, 0.925)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for i, (mean, ci) in enumerate(zip(means, cis)):
        ax.text(
            i,
            mean + ci + 0.001,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    add_bracket(ax, 0, 2, 0.919, 0.001, "p=0.002 **")
    add_bracket(ax, 1, 3, 0.922, 0.001, "p=0.052")

    fig.tight_layout()
    save(fig, "01_main_result_bar")
    print("✓ Plot 1: Main result bar chart")

    # PLOT 2: Val accuracy curves over epochs
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, 6)
    for group_name in GROUP_ORDER:
        arr = np.array([run["val_accuracies"] for run in groups[group_name]])
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.plot(
            epochs,
            mean,
            "o-",
            color=COLORS[group_name],
            label=SHORT_NAMES[group_name],
            linewidth=2,
            markersize=5,
            zorder=3,
        )
        ax.fill_between(
            epochs,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.15,
            color=COLORS[group_name],
            zorder=2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy Over Training (mean +- 95% CI)")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, fancybox=True, shadow=False, edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save(fig, "02_val_accuracy_curves")
    print("✓ Plot 2: Val accuracy curves")

    # PLOT 3: Training loss curves over epochs
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group_name in GROUP_ORDER:
        arr = np.array([run["train_losses"] for run in groups[group_name]])
        mean = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.plot(
            epochs,
            mean,
            "s-",
            color=COLORS[group_name],
            label=SHORT_NAMES[group_name],
            linewidth=2,
            markersize=5,
            zorder=3,
        )
        ax.fill_between(
            epochs,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.15,
            color=COLORS[group_name],
            zorder=2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Over Epochs (mean +- 95% CI)")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, fancybox=True, shadow=False, edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    save(fig, "03_train_loss_curves")
    print("✓ Plot 3: Training loss curves")

    # PLOT 4: Strip/swarm + box plot of best val acc
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data_for_box = []
    positions = []
    for i, group_name in enumerate(GROUP_ORDER):
        vals = [run["best_val_accuracy"] for run in groups[group_name]]
        data_for_box.append(vals)
        positions.append(i)

    bp = ax.boxplot(
        data_for_box,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
        ),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, group_name in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(COLORS[group_name])
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    rng = np.random.RandomState(42)
    for i, group_name in enumerate(GROUP_ORDER):
        vals = [run["best_val_accuracy"] for run in groups[group_name]]
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            color=COLORS[group_name],
            edgecolor="black",
            s=50,
            zorder=5,
            linewidth=0.8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER])
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Distribution of Best Val Accuracy Across Seeds")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save(fig, "04_boxplot_best_val")
    print("✓ Plot 4: Box + strip plot")

    # PLOT 5: 2x2 Factorial interaction plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    schedulers = ["No scheduler", "Warmup linear"]
    for method, marker, color in [
        ("AdamW", "s", "#3498db"),
        ("Adaptive (ours)", "o", "#e74c3c"),
    ]:
        y_vals = []
        y_errs = []
        for sched_suffix in ["", " + warmup linear"]:
            key = method + sched_suffix if sched_suffix else method
            vals = [run["best_val_accuracy"] for run in groups[key]]
            y_vals.append(np.mean(vals))
            y_errs.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        label = "AdamW" if "AdamW" in method else "Adaptive (ours)"
        ax.errorbar(
            [0, 1],
            y_vals,
            yerr=y_errs,
            fmt=f"{marker}-",
            color=color,
            linewidth=2.5,
            markersize=10,
            capsize=6,
            label=label,
            markeredgecolor="black",
            markeredgewidth=0.8,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(schedulers, fontsize=12)
    ax.set_ylabel("Mean Best Validation Accuracy")
    ax.set_title("Interaction: Optimizer x LR Scheduler")
    ax.legend(frameon=True, fancybox=True, edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    adamw_delta = np.mean(
        [run["best_val_accuracy"] for run in groups["AdamW + warmup linear"]]
    ) - np.mean([run["best_val_accuracy"] for run in groups["AdamW"]])
    algo_delta = np.mean(
        [
            run["best_val_accuracy"]
            for run in groups["Adaptive (ours) + warmup linear"]
        ]
    ) - np.mean([run["best_val_accuracy"] for run in groups["Adaptive (ours)"]])
    ax.annotate(
        f"Delta = +{adamw_delta:.4f}",
        xy=(
            0.5,
            np.mean(
                [
                    np.mean([run["best_val_accuracy"] for run in groups["AdamW"]]),
                    np.mean(
                        [
                            run["best_val_accuracy"]
                            for run in groups["AdamW + warmup linear"]
                        ]
                    ),
                ]
            ),
        ),
        fontsize=10,
        color="#3498db",
        ha="center",
        fontweight="bold",
    )
    ax.annotate(
        f"Delta = +{algo_delta:.4f}",
        xy=(
            0.5,
            np.mean(
                [
                    np.mean(
                        [run["best_val_accuracy"] for run in groups["Adaptive (ours)"]]
                    ),
                    np.mean(
                        [
                            run["best_val_accuracy"]
                            for run in groups["Adaptive (ours) + warmup linear"]
                        ]
                    ),
                ]
            ),
        ),
        fontsize=10,
        color="#e74c3c",
        ha="center",
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "05_interaction_plot")
    print("✓ Plot 5: 2x2 interaction plot")

    # PLOT 6: Train-Val gap analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for group_name in GROUP_ORDER:
        train_accs = [run["train_accuracies"][-1] for run in groups[group_name]]
        val_accs = [run["final_val_accuracy"] for run in groups[group_name]]
        ax.scatter(
            train_accs,
            val_accs,
            color=COLORS[group_name],
            edgecolor="black",
            s=80,
            label=SHORT_NAMES[group_name],
            zorder=5,
            linewidth=0.8,
        )

    ax.plot([0.96, 1.0], [0.96, 1.0], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Final Training Accuracy")
    ax.set_ylabel("Final Validation Accuracy")
    ax.set_title("Generalization Gap")
    ax.legend(frameon=True, fancybox=True, edgecolor="gray", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    gaps = []
    gap_errs = []
    for group_name in GROUP_ORDER:
        g_vals = [
            run["train_accuracies"][-1] - run["final_val_accuracy"]
            for run in groups[group_name]
        ]
        gaps.append(np.mean(g_vals))
        gap_errs.append(1.96 * np.std(g_vals, ddof=1) / np.sqrt(len(g_vals)))

    bars = ax.bar(
        range(len(GROUP_ORDER)),
        gaps,
        yerr=gap_errs,
        color=[COLORS[group_name] for group_name in GROUP_ORDER],
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        width=0.6,
    )
    for i, group_name in enumerate(GROUP_ORDER):
        bars[i].set_hatch(HATCHES[group_name])
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER], fontsize=10)
    ax.set_ylabel("Train - Val Accuracy Gap")
    ax.set_title("Overfitting: Train-Val Gap (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    for i, (gap, err) in enumerate(zip(gaps, gap_errs)):
        ax.text(
            i,
            gap + err + 0.002,
            f"{gap:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    save(fig, "06_generalization_gap")
    print("✓ Plot 6: Generalization gap")

    # PLOT 7: Validation accuracy degradation (best->final)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, group_name in enumerate(GROUP_ORDER):
        best_vals = [run["best_val_accuracy"] for run in groups[group_name]]
        final_vals = [run["final_val_accuracy"] for run in groups[group_name]]
        for j, (best_val, final_val) in enumerate(zip(best_vals, final_vals)):
            jitter = (j - 2) * 0.06
            ax.plot(
                [i + jitter] * 2,
                [best_val, final_val],
                color=COLORS[group_name],
                linewidth=1.5,
                alpha=0.6,
            )
            ax.scatter(
                i + jitter,
                best_val,
                color=COLORS[group_name],
                marker="^",
                s=40,
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )
            ax.scatter(
                i + jitter,
                final_val,
                color=COLORS[group_name],
                marker="v",
                s=40,
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
            )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            label="Best val",
            markersize=8,
            linestyle="None",
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="gray",
            label="Final val",
            markersize=8,
            linestyle="None",
            markeredgecolor="black",
        ),
    ]
    ax.legend(handles=legend_elements, frameon=True, fancybox=True, edgecolor="gray")
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER])
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Best vs. Final Validation Accuracy per Seed")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save(fig, "07_best_vs_final_val")
    print("✓ Plot 7: Best vs final val")

    # PLOT 8: Episode reward trajectories (algo runs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, group_name, title in zip(
        axes,
        ["Adaptive (ours)", "Adaptive (ours) + warmup linear"],
        ["Adaptive (no scheduler)", "Adaptive + warmup linear"],
    ):
        for run in groups[group_name]:
            episode_logs = run.get("episode_logs")
            if episode_logs and "episode_rewards" in episode_logs:
                rewards = episode_logs["episode_rewards"]
                episodes = np.arange(len(rewards))
                ax.plot(episodes, rewards, alpha=0.25, color=COLORS[group_name], linewidth=0.8)

        all_rewards = []
        for run in groups[group_name]:
            episode_logs = run.get("episode_logs")
            if episode_logs and "episode_rewards" in episode_logs:
                all_rewards.append(episode_logs["episode_rewards"])
        if all_rewards:
            min_len = min(len(rewards) for rewards in all_rewards)
            arr = np.array([rewards[:min_len] for rewards in all_rewards])
            mean = arr.mean(axis=0)
            window = 5
            smoothed = np.convolve(mean, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(len(smoothed)) + window // 2,
                smoothed,
                color="black",
                linewidth=2,
                label="Mean (smoothed)",
            )
        ax.set_xlabel("Episode")
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, edgecolor="gray")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Episode Reward")
    fig.suptitle("Bandit Reward Trajectory Over Training", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "08_episode_rewards")
    print("✓ Plot 8: Episode reward trajectories")

    # PLOT 9: LR multiplier selection over episodes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    lr_colors = {0.7: "#3498db", 1.0: "#95a5a6", 1.3: "#e74c3c"}

    for ax, group_name, title in zip(
        axes,
        ["Adaptive (ours)", "Adaptive (ours) + warmup linear"],
        ["Adaptive (no scheduler)", "Adaptive + warmup linear"],
    ):
        all_lr_selections = []
        for run in groups[group_name]:
            episode_logs = run.get("episode_logs")
            if episode_logs and "selected_lr_values" in episode_logs:
                all_lr_selections.append(episode_logs["selected_lr_values"])

        if all_lr_selections:
            min_len = min(len(selection) for selection in all_lr_selections)
            arr = np.array([selection[:min_len] for selection in all_lr_selections])
            for lr_val, color in lr_colors.items():
                frac = (arr == lr_val).mean(axis=0)
                window = 5
                smoothed = np.convolve(frac, np.ones(window) / window, mode="valid")
                ax.plot(
                    np.arange(len(smoothed)) + window // 2,
                    smoothed,
                    color=color,
                    linewidth=2,
                    label=f"LRx{lr_val}",
                )
        ax.set_xlabel("Episode")
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, edgecolor="gray")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Selection Frequency")
    fig.suptitle("LR Multiplier Selection Probability Over Training", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "09_lr_selection_over_time")
    print("✓ Plot 9: LR multiplier selection over time")

    # PLOT 10: Context trend distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    trend_colors = {
        "improving": "#2ecc71",
        "stable": "#f39c12",
        "worsening": "#e74c3c",
    }

    for ax, group_name, title in zip(
        axes,
        ["Adaptive (ours)", "Adaptive (ours) + warmup linear"],
        ["Adaptive (no scheduler)", "Adaptive + warmup linear"],
    ):
        all_trends = []
        all_steps = []
        for run in groups[group_name]:
            episode_logs = run.get("episode_logs")
            if (
                episode_logs
                and "context_trends" in episode_logs
                and "episode_end_steps" in episode_logs
            ):
                all_trends.extend(episode_logs["context_trends"])
                all_steps.extend(episode_logs["episode_end_steps"])

        if all_trends:
            max_step = max(all_steps)
            n_bins = 5
            bin_edges = np.linspace(0, max_step, n_bins + 1)
            bin_labels = [f"Epoch {i + 1}" for i in range(n_bins)]

            bottom_improving = []
            bottom_stable = []
            bottom_worsening = []
            for b in range(n_bins):
                mask = [(step > bin_edges[b]) & (step <= bin_edges[b + 1]) for step in all_steps]
                trends_in_bin = [trend for trend, include in zip(all_trends, mask) if include]
                total = max(len(trends_in_bin), 1)
                counts = Counter(trends_in_bin)
                bottom_improving.append(counts.get("improving", 0) / total)
                bottom_stable.append(counts.get("stable", 0) / total)
                bottom_worsening.append(counts.get("worsening", 0) / total)

            x = np.arange(n_bins)
            ax.bar(
                x,
                bottom_improving,
                0.6,
                label="Improving",
                color=trend_colors["improving"],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                x,
                bottom_stable,
                0.6,
                bottom=bottom_improving,
                label="Stable",
                color=trend_colors["stable"],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                x,
                bottom_worsening,
                0.6,
                bottom=[improving + stable for improving, stable in zip(bottom_improving, bottom_stable)],
                label="Worsening",
                color=trend_colors["worsening"],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, fontsize=9)
            ax.set_title(title)
            ax.legend(frameon=True, fancybox=True, edgecolor="gray", fontsize=9)
            ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Proportion of Episodes")
    fig.suptitle("Loss Trend Context Distribution Over Training", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "10_context_trends")
    print("✓ Plot 10: Context trend distribution")

    # PLOT 11: Pairwise significance heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    n = len(GROUP_ORDER)
    pval_matrix = np.ones((n, n))
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                a = [run["best_val_accuracy"] for run in groups[GROUP_ORDER[i]]]
                b = [run["best_val_accuracy"] for run in groups[GROUP_ORDER[j]]]
                _, p = stats.ttest_ind(a, b, equal_var=False)
                pval_matrix[i, j] = p
                diff_matrix[i, j] = np.mean(a) - np.mean(b)

    cmap = LinearSegmentedColormap.from_list("sig", ["#2ecc71", "#f1c40f", "#e74c3c"], N=256)

    im = ax.imshow(pval_matrix, cmap=cmap, vmin=0, vmax=0.1, aspect="equal")
    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)

    short_labels = [SHORT_NAMES[group_name] for group_name in GROUP_ORDER]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, color="black")
            else:
                p = pval_matrix[i, j]
                diff = diff_matrix[i, j]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                text_color = "white" if p < 0.03 else "black"
                ax.text(
                    j,
                    i,
                    f"{diff:+.4f}\n({sig})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_title("Pairwise Differences in Best Val Accuracy\n(row - column, Welch's t-test)")
    fig.tight_layout()
    save(fig, "11_significance_heatmap")
    print("✓ Plot 11: Significance heatmap")

    # PLOT 12: Per-seed paired comparison (slope chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    seed_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for g_idx, group_name in enumerate(GROUP_ORDER):
        for run in groups[group_name]:
            seed = run["seed"]
            ax.scatter(
                g_idx,
                run["best_val_accuracy"],
                color=seed_colors[seed],
                s=60,
                zorder=5,
                edgecolor="black",
                linewidth=0.5,
            )

    for seed_idx in range(5):
        ys = []
        for group_name in GROUP_ORDER:
            for run in groups[group_name]:
                if run["seed"] == seed_idx:
                    ys.append(run["best_val_accuracy"])
        ax.plot(
            range(len(GROUP_ORDER)),
            ys,
            color=seed_colors[seed_idx],
            alpha=0.4,
            linewidth=1.2,
            linestyle="--",
            label=f"Seed {seed_idx}" if seed_idx < 5 else None,
        )

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER])
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Per-Seed Paired Comparison Across Configurations")
    ax.legend(frameon=True, fancybox=True, edgecolor="gray", ncol=5, fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    save(fig, "12_paired_seed_comparison")
    print("✓ Plot 12: Paired seed comparison")

    # PLOT 13: Performance Delta - Adaptive over AdamW (paired by seed)
    fig, ax = plt.subplots(figsize=(6, 5))

    scheduler_pairs = [
        ("No Scheduler", "AdamW", "Adaptive (ours)"),
        ("Warmup Linear", "AdamW + warmup linear", "Adaptive (ours) + warmup linear"),
    ]
    bar_colors = ["#e74c3c", "#c0392b"]
    x_pos = np.arange(len(scheduler_pairs))

    deltas_all = []
    sems_all = []
    for _, adamw_key, algo_key in scheduler_pairs:
        adamw_by_seed = by_seed(adamw_key)
        algo_by_seed = by_seed(algo_key)
        paired_deltas = []
        for seed in sorted(adamw_by_seed.keys()):
            delta = algo_by_seed[seed]["best_val_accuracy"] - adamw_by_seed[seed]["best_val_accuracy"]
            paired_deltas.append(delta * 100)
        deltas_all.append(np.mean(paired_deltas))
        sems_all.append(np.std(paired_deltas, ddof=1) / np.sqrt(len(paired_deltas)))

    bars = ax.bar(
        x_pos,
        deltas_all,
        yerr=sems_all,
        width=0.5,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=6,
        error_kw={"linewidth": 1.5},
        zorder=3,
    )

    for i, (delta, sem) in enumerate(zip(deltas_all, sems_all)):
        ax.text(
            i,
            delta + sem + 0.02,
            f"+{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pair[0] for pair in scheduler_pairs], fontsize=12)
    ax.set_ylabel("Adaptive - AdamW (pp)")
    ax.set_title("Improvement from Adaptive Wrapper over AdamW\n(Best Val Accuracy, paired by seed +- SEM)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    save(fig, "13_performance_delta")
    print("✓ Plot 13: Performance delta")

    # PLOT 14: Heatmap - Optimizer x Scheduler grid
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    optimizers = ["AdamW", "Adaptive (ours)"]
    schedulers_map = {
        "No Scheduler": ("AdamW", "Adaptive (ours)"),
        "Warmup Linear": ("AdamW + warmup linear", "Adaptive (ours) + warmup linear"),
    }
    sched_labels = ["No Scheduler", "Warmup Linear"]

    data_grid = np.zeros((2, 2))
    annot_grid = []
    for i, _ in enumerate(optimizers):
        row_annot = []
        for j, sched in enumerate(sched_labels):
            key = schedulers_map[sched][i]
            vals = [run["best_val_accuracy"] * 100 for run in groups[key]]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            data_grid[i, j] = mean
            row_annot.append(f"{mean:.2f}\n+-{std:.2f}")
        annot_grid.append(row_annot)

    cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c", "#f1c40f", "#2ecc71"], N=256)

    im = ax.imshow(
        data_grid,
        cmap=cmap,
        aspect="auto",
        vmin=data_grid.min() - 0.3,
        vmax=data_grid.max() + 0.3,
    )
    plt.colorbar(im, ax=ax, shrink=0.9, label="Best Val Acc (%)")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                annot_grid[i][j],
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="black",
            )

    ax.set_xticks(range(2))
    ax.set_xticklabels(sched_labels, fontsize=11)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["AdamW", "Adaptive\n(ours)"], fontsize=11)
    ax.set_title("Best Val Accuracy (%) - SST-2")
    fig.tight_layout()
    save(fig, "14_heatmap_grid")
    print("✓ Plot 14: Heatmap grid")

    # PLOT 15: Best vs Final Val Accuracy side-by-side bars
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(GROUP_ORDER))
    width = 0.35

    best_means, best_cis = [], []
    final_means, final_cis = [], []
    for group_name in GROUP_ORDER:
        best_vals = [run["best_val_accuracy"] * 100 for run in groups[group_name]]
        final_vals = [run["final_val_accuracy"] * 100 for run in groups[group_name]]
        best_means.append(np.mean(best_vals))
        best_cis.append(1.96 * np.std(best_vals, ddof=1) / np.sqrt(len(best_vals)))
        final_means.append(np.mean(final_vals))
        final_cis.append(1.96 * np.std(final_vals, ddof=1) / np.sqrt(len(final_vals)))

    bar_colors_best = [COLORS[group_name] for group_name in GROUP_ORDER]
    bar_colors_final = [COLORS[group_name] for group_name in GROUP_ORDER]

    ax.bar(
        x - width / 2,
        best_means,
        width,
        yerr=best_cis,
        label="Best Val Acc",
        color=bar_colors_best,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.2},
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        final_means,
        width,
        yerr=final_cis,
        label="Final Val Acc",
        color=bar_colors_final,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.2},
        zorder=3,
        hatch="///",
        alpha=0.7,
    )

    for i, (best_mean, best_ci) in enumerate(zip(best_means, best_cis)):
        ax.text(
            i - width / 2,
            best_mean + best_ci + 0.1,
            f"{best_mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER], fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("SST-2 - Best vs Final Validation Accuracy")
    ax.legend(frameon=True, fancybox=True, edgecolor="gray", loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(86, 93)
    fig.tight_layout()
    save(fig, "15_best_vs_final_bars")
    print("✓ Plot 15: Best vs Final bars")

    # PLOT 16: Per-Seed Strip Plot (CIFAR-100 style)
    fig, ax = plt.subplots(figsize=(8, 5))

    adamw_color = "#3498db"
    algo_color = "#e74c3c"
    method_colors = {
        "AdamW": adamw_color,
        "AdamW + warmup linear": adamw_color,
        "Adaptive (ours)": algo_color,
        "Adaptive (ours) + warmup linear": algo_color,
    }

    for i, group_name in enumerate(GROUP_ORDER):
        vals = [run["best_val_accuracy"] * 100 for run in groups[group_name]]
        mean_val = np.mean(vals)

        rng = np.random.RandomState(42 + i)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            color=method_colors[group_name],
            alpha=0.6,
            edgecolor="black",
            s=70,
            linewidth=0.6,
            zorder=5,
        )

        ax.hlines(mean_val, i - 0.25, i + 0.25, color=method_colors[group_name], linewidth=3, zorder=6)

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([SHORT_NAMES[group_name] for group_name in GROUP_ORDER], fontsize=11)
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("Best Val Accuracy by Seed - SST-2")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    legend_elements = [
        Line2D([0], [0], color=adamw_color, linewidth=3, label="AdamW (mean)"),
        Line2D([0], [0], color=algo_color, linewidth=3, label="Adaptive (mean)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label="Individual seed",
        ),
    ]
    ax.legend(handles=legend_elements, frameon=True, fancybox=True, edgecolor="gray")
    fig.tight_layout()
    save(fig, "16_strip_plot_seeds")
    print("✓ Plot 16: Strip plot")

    # PLOT 17: Stability - rolling epoch-level std of val accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, 6)

    for group_name in GROUP_ORDER:
        arr = np.array([run["val_accuracies"] for run in groups[group_name]]) * 100
        std_per_epoch = arr.std(axis=0, ddof=1)
        ax.plot(
            epochs,
            std_per_epoch,
            "o-",
            color=COLORS[group_name],
            label=SHORT_NAMES[group_name],
            linewidth=2,
            markersize=7,
            zorder=3,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std Dev Across Seeds (pp)")
    ax.set_title("Validation Accuracy Stability - Cross-Seed Std Dev per Epoch")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, fancybox=True, edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    save(fig, "17_stability_per_epoch")
    print("✓ Plot 17: Stability per epoch")

    # PLOT 18: Generalization gap over epochs (train - val per epoch)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, 6)

    for group_name in GROUP_ORDER:
        train_arr = np.array([run["train_accuracies"] for run in groups[group_name]]) * 100
        val_arr = np.array([run["val_accuracies"] for run in groups[group_name]]) * 100
        gap_arr = train_arr - val_arr
        mean = gap_arr.mean(axis=0)
        se = gap_arr.std(axis=0, ddof=1) / np.sqrt(gap_arr.shape[0])
        ax.plot(
            epochs,
            mean,
            "s-",
            color=COLORS[group_name],
            label=SHORT_NAMES[group_name],
            linewidth=2,
            markersize=6,
            zorder=3,
        )
        ax.fill_between(
            epochs,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.15,
            color=COLORS[group_name],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train - Val Accuracy (pp)")
    ax.set_title("Generalization Gap Over Training (mean +- 95% CI)")
    ax.set_xticks(epochs)
    ax.legend(frameon=True, fancybox=True, edgecolor="gray")
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    save(fig, "18_gen_gap_over_epochs")
    print("✓ Plot 18: Gen gap over epochs")

    # EXTRA PLOT: Phase1b follow-up best val accuracy bar chart (non-AdamW only)
    fig, ax = plt.subplots(figsize=(8, 5))
    phase1b_order = sorted(phase1b_followup_groups)
    means = []
    stds = []
    labels = []
    colors = ["#e74c3c", "#ff9896", "#f39c12", "#2ecc71"]

    for group_name in phase1b_order:
        vals = [run["best_val_accuracy"] for run in phase1b_followup_groups[group_name]]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(short_name_phase1b_followup(group_name))

    x = np.arange(len(means))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(means)],
        edgecolor="black",
        linewidth=0.5,
        width=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("SST-2 Phase1b Follow-up - Best Val Accuracy (mean +- std, 5 seeds)", fontsize=13)
    ax.set_ylim(0.85, 0.92)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i,
            mean + std + 0.001,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()
    save(fig, "best_val_phase1b")
    print("✓ Extra plot: Phase1b follow-up best val bar")

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
