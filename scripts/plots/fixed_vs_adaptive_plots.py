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

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "fixed_vs_adaptive_none"
OUTDIR = ROOT / "results" / "plots" / "fixed_vs_adaptive"

SEEDS = list(range(5))

C_ADAPTIVE = "#2166ac"
C_FIXED = "#b2182b"
C_A_FILL = "#92c5de"
C_F_FILL = "#f4a582"


def apply_base_rcparams():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
        }
    )


def apply_thesis_rcparams():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
        }
    )


def save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.png")
    fig.savefig(OUTDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + .pdf")


def load_all():
    data = {}
    for json_path in sorted(BASE.glob("seed_*/*.json")):
        seed_match = re.search(r"seed_(\d+)", str(json_path.parent))
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        with open(json_path, encoding="utf-8") as fh:
            run = json.load(fh)

        fname = json_path.name
        if "sst2" in fname:
            category = "sst2_adaptive" if "main" in fname else "sst2_fixed"
        elif "cifar_clean" in fname:
            category = "cifar_clean_adaptive" if "aees" in fname else "cifar_clean_fixed"
        elif "cifar_noisy" in fname:
            category = "cifar_noisy_adaptive" if "aees" in fname else "cifar_noisy_fixed"
        else:
            continue

        data.setdefault(category, {})[seed] = run

    required = [
        "sst2_adaptive",
        "sst2_fixed",
        "cifar_clean_adaptive",
        "cifar_clean_fixed",
        "cifar_noisy_adaptive",
        "cifar_noisy_fixed",
    ]
    missing = []
    for key in required:
        if key not in data:
            missing.append(key)
            continue
        for seed in SEEDS:
            if seed not in data[key]:
                missing.append(f"{key} seed {seed}")
    if missing:
        raise FileNotFoundError("Missing fixed-vs-adaptive runs:\n" + "\n".join(missing))

    return data


def get_vals(all_data, key, field):
    return [all_data[key][seed][field] for seed in SEEDS]


def plot2b_training_curves_sst2(all_data):
    apply_base_rcparams()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, color, light, label in [
        ("fixed", C_FIXED, C_F_FILL, "Fixed"),
        ("adaptive", C_ADAPTIVE, C_A_FILL, "Adaptive"),
    ]:
        key = f"sst2_{method}"
        curves = np.array([all_data[key][seed]["val_accuracies"] for seed in SEEDS])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=1)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, color=color, label=label, zorder=3, marker="o", markersize=5)
        ax.fill_between(epochs, mean - std, mean + std, color=light, alpha=0.35, zorder=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("SST-2 - Validation Accuracy Curves")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    save(fig, "plot2b_training_curves_sst2")
    print("Plot 2b done")


def plot3_bandit_dynamics(all_data):
    apply_base_rcparams()
    arm_colors = {0.5: "#2166ac", 1.0: "#4dac26", 2.0: "#d01c8b"}
    arm_labels = {0.5: r"$\alpha=0.5$", 1.0: r"$\alpha=1.0$", 2.0: r"$\alpha=2.0$"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (task, title) in enumerate([("cifar_clean", "CIFAR-100 (Clean)"), ("cifar_noisy", "CIFAR-100 (Noisy)")]):
        run = all_data[f"{task}_adaptive"][0]
        lr_logs = run["controller_logs"]["lr_controller_logs"]
        value_hist = lr_logs["value_estimates_history"]
        count_hist = lr_logs["warmup_counts_history"]
        steps = np.arange(len(value_hist))
        arms = [0.5, 1.0, 2.0]

        ax = axes[0, col]
        for arm in arms:
            vals = [entry[str(arm)] for entry in value_hist]
            ax.plot(steps, vals, color=arm_colors[arm], label=arm_labels[arm], alpha=0.85)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Reward Value Estimate")
        ax.set_title(f"{title} - Bandit Dynamics (seed 0)")
        ax.legend(loc="lower left")
        ax.set_xlim(0, len(steps))

        ax2 = axes[1, col]
        for arm in arms:
            counts = [entry[str(arm)] for entry in count_hist]
            ax2.plot(steps, counts, color=arm_colors[arm], label=arm_labels[arm], alpha=0.85)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Cumulative Arm Pulls")
        ax2.legend(loc="upper left")
        ax2.set_xlim(0, len(steps))

    fig.suptitle("UCB Bandit Controller - Value Estimates & Arm Selection", y=1.01, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "plot3_bandit_dynamics")
    print("Plot 3 done")


def plot4_seed_paired_slopes(all_data):
    apply_base_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    tasks = [("cifar_clean", "CIFAR-100 (Clean)"), ("cifar_noisy", "CIFAR-100 (Noisy)"), ("sst2", "SST-2")]

    for idx, (task, title) in enumerate(tasks):
        ax = axes[idx]
        for seed in SEEDS:
            fixed_val = all_data[f"{task}_fixed"][seed]["best_val_accuracy"]
            adaptive_val = all_data[f"{task}_adaptive"][seed]["best_val_accuracy"]
            color = "#1a6b1a" if adaptive_val > fixed_val else "#b2182b"
            ax.plot([0, 1], [fixed_val, adaptive_val], "o-", color=color, alpha=0.7, markersize=7, linewidth=1.8)
            ax.annotate(f"s{seed}", xy=(1.03, adaptive_val), fontsize=8, va="center", color="#555")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fixed", "Adaptive"], fontweight="bold")
        ax.set_ylabel("Best Validation Accuracy")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
        ax.set_xlim(-0.15, 1.25)

    fig.suptitle("Seed-Paired Comparison: Fixed -> Adaptive", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "plot4_seed_paired_slopes")
    print("Plot 4 done")


def plot8_training_loss(all_data):
    apply_base_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for col, (task, title) in enumerate([("cifar_clean", "CIFAR-100 (Clean)"), ("cifar_noisy", "CIFAR-100 (Noisy)")]):
        ax = axes[col]
        for method, color, light, label in [
            ("fixed", C_FIXED, C_F_FILL, "Fixed"),
            ("adaptive", C_ADAPTIVE, C_A_FILL, "Adaptive"),
        ]:
            key = f"{task}_{method}"
            curves = np.array([np.array(all_data[key][seed]["train_losses"]) for seed in SEEDS])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=color, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=light, alpha=0.3)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_xlim(1, 200)
        ax.set_yscale("log")

    fig.suptitle("Training Loss Curves (log scale, mean +- 1 std)", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "plot8_training_loss")
    print("Plot 8 done")


def fig1_summary_bar(all_data):
    apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    tasks = ["CIFAR-100 (Clean)", "CIFAR-100 (Noisy)", "SST-2"]
    task_keys = ["cifar_clean", "cifar_noisy", "sst2"]

    adaptive_means, adaptive_stds, fixed_means, fixed_stds = [], [], [], []
    for task_key in task_keys:
        adaptive_vals = get_vals(all_data, f"{task_key}_adaptive", "best_val_accuracy")
        fixed_vals = get_vals(all_data, f"{task_key}_fixed", "best_val_accuracy")
        adaptive_means.append(np.mean(adaptive_vals))
        adaptive_stds.append(np.std(adaptive_vals, ddof=1))
        fixed_means.append(np.mean(fixed_vals))
        fixed_stds.append(np.std(fixed_vals, ddof=1))

    x = np.arange(len(tasks))
    width = 0.30

    bars_fixed = ax.bar(
        x - width / 2,
        fixed_means,
        width,
        yerr=fixed_stds,
        label="Fixed",
        color=C_F_FILL,
        edgecolor=C_FIXED,
        linewidth=1.1,
        capsize=3.5,
        error_kw={"linewidth": 1.0},
    )
    bars_adaptive = ax.bar(
        x + width / 2,
        adaptive_means,
        width,
        yerr=adaptive_stds,
        label="Adaptive",
        color=C_A_FILL,
        edgecolor=C_ADAPTIVE,
        linewidth=1.1,
        capsize=3.5,
        error_kw={"linewidth": 1.0},
    )

    for bars, means in [(bars_fixed, fixed_means), (bars_adaptive, adaptive_means)]:
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{mean:.1%}", ha="center", va="bottom", fontsize=8.5, fontweight="medium")

    for i in range(3):
        delta = (adaptive_means[i] - fixed_means[i]) * 100
        y_top = max(adaptive_means[i] + adaptive_stds[i], fixed_means[i] + fixed_stds[i]) + 0.025
        sign = "+" if delta > 0 else ""
        color = "#1a7a1a" if delta > 0.1 else "#666666"
        ax.annotate(rf"$\Delta$={sign}{delta:.1f} pp", xy=(x[i], y_top), ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel("Mean Best Validation Accuracy (5 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(loc="lower left")
    ax.set_ylim(0.54, 0.97)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Adaptive vs. Best Fixed Candidate")
    ax.yaxis.grid(True)
    save(fig, "fig1_summary_bar")
    print("Fig 1 done")


def fig2_best_vs_final(all_data):
    apply_thesis_rcparams()
    tasks = ["CIFAR-100 (Clean)", "CIFAR-100 (Noisy)", "SST-2"]
    task_keys = ["cifar_clean", "cifar_noisy", "sst2"]
    ylims = [(0.66, 0.74), (0.51, 0.65), (0.87, 0.93)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for idx, (task_key, title, (ylo, yhi)) in enumerate(zip(task_keys, tasks, ylims)):
        ax = axes[idx]
        groups = [
            ("Adaptive\nbest", get_vals(all_data, f"{task_key}_adaptive", "best_val_accuracy"), C_A_FILL, C_ADAPTIVE),
            ("Adaptive\nfinal", get_vals(all_data, f"{task_key}_adaptive", "final_val_accuracy"), "#c6dbef", C_ADAPTIVE),
            ("Fixed\nbest", get_vals(all_data, f"{task_key}_fixed", "best_val_accuracy"), C_F_FILL, C_FIXED),
            ("Fixed\nfinal", get_vals(all_data, f"{task_key}_fixed", "final_val_accuracy"), "#fdd0a2", C_FIXED),
        ]
        positions = [0, 0.75, 1.8, 2.55]

        for pos, (_, vals, color, edge) in zip(positions, groups):
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            ax.bar(pos, mean, 0.6, yerr=std, color=color, edgecolor=edge, linewidth=1.0, capsize=3, error_kw={"linewidth": 0.9}, zorder=3)
            ax.text(pos, mean + std + (yhi - ylo) * 0.02, f"{mean:.1%}", ha="center", va="bottom", fontsize=8)

        for best_idx, final_idx, edge in [(0, 1, C_ADAPTIVE), (2, 3, C_FIXED)]:
            best_mean = np.mean(groups[best_idx][1])
            final_mean = np.mean(groups[final_idx][1])
            drop = (best_mean - final_mean) * 100
            mid_x = (positions[best_idx] + positions[final_idx]) / 2
            ax.annotate("", xy=(mid_x, final_mean), xytext=(mid_x, best_mean), arrowprops=dict(arrowstyle="->", color=edge, lw=1.3, ls="--"))
            ax.text(mid_x + 0.22, (best_mean + final_mean) / 2, f"−{drop:.1f}pp", fontsize=8, color=edge, va="center", fontstyle="italic")

        ax.set_xticks(positions)
        ax.set_xticklabels([group[0] for group in groups], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.grid(True)
        if idx == 0:
            ax.set_ylabel("Validation Accuracy")

    fig.suptitle("Best vs. Final Validation Accuracy", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_best_vs_final")
    print("Fig 2 done")


def fig3_per_seed_delta(all_data):
    apply_thesis_rcparams()
    tasks = ["CIFAR-100 (Clean)", "CIFAR-100 (Noisy)", "SST-2"]
    task_keys = ["cifar_clean", "cifar_noisy", "sst2"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=False)
    for idx, (task_key, title) in enumerate(zip(task_keys, tasks)):
        ax = axes[idx]
        deltas = []
        for seed in SEEDS:
            adaptive_val = all_data[f"{task_key}_adaptive"][seed]["best_val_accuracy"]
            fixed_val = all_data[f"{task_key}_fixed"][seed]["best_val_accuracy"]
            deltas.append((adaptive_val - fixed_val) * 100)

        mean_delta = np.mean(deltas)
        seed_x = np.arange(len(SEEDS))
        ax.axhline(0, color="0.4", linewidth=0.8, linestyle="-", zorder=1)
        colors = [C_ADAPTIVE if delta > 0 else C_FIXED for delta in deltas]
        ax.bar(seed_x, deltas, width=0.55, color=colors, alpha=0.7, edgecolor=colors, linewidth=1.0, zorder=3)
        ax.axhline(mean_delta, color="black", linewidth=1.3, linestyle="--", zorder=4, label=f"Mean Δ = {mean_delta:+.2f} pp")
        ax.set_xticks(seed_x)
        ax.set_xticklabels([f"Seed {seed}" for seed in SEEDS], fontsize=9)
        ax.set_ylabel("Δ Best Val Acc (pp)" if idx == 0 else "")
        ax.set_title(title)
        ax.legend(loc="upper right" if mean_delta > 0 else "lower right", fontsize=8.5)
        ax.yaxis.grid(True)

        wins = sum(1 for delta in deltas if delta > 0)
        ax.text(0.03, 0.03, f"{wins}/5 seeds positive", transform=ax.transAxes, fontsize=8, color="0.35", va="bottom")

    fig.suptitle("Per-Seed Accuracy Gain: Adaptive − Fixed (pp)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_per_seed_delta")
    print("Fig 3 done")


def fig4_cifar_val_curves(all_data):
    apply_thesis_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for col, (task_key, title) in enumerate([("cifar_clean", "CIFAR-100 (Clean)"), ("cifar_noisy", "CIFAR-100 (Noisy Labels)")]):
        ax = axes[col]
        for method, color, fill, label in [
            ("fixed", C_FIXED, C_F_FILL, "Fixed"),
            ("adaptive", C_ADAPTIVE, C_A_FILL, "Adaptive"),
        ]:
            key = f"{task_key}_{method}"
            curves = np.array([all_data[key][seed]["val_accuracies"] for seed in SEEDS])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=color, label=label, zorder=3)
            ax.fill_between(epochs, mean - std, mean + std, color=fill, alpha=0.28, zorder=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy" if col == 0 else "")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(1, 200)
        ax.yaxis.grid(True)

    fig.suptitle("Validation Accuracy: Adaptive vs. Best Fixed Candidate (mean ± 1 std, 5 seeds)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_cifar_val_curves")
    print("Fig 4 done")


def fig5_controller_arm_selection(all_data):
    apply_thesis_rcparams()
    arm_colors = {0.5: "#2166ac", 1.0: "#4dac26", 2.0: "#d01c8b"}
    arm_labels = {0.5: r"$\alpha\!=\!0.5$", 1.0: r"$\alpha\!=\!1.0$", 2.0: r"$\alpha\!=\!2.0$"}
    arms = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for col, (task_key, title) in enumerate([("cifar_clean", "CIFAR-100 (Clean)"), ("cifar_noisy", "CIFAR-100 (Noisy Labels)")]):
        ax = axes[col]
        all_fracs = {arm: [] for arm in arms}
        final_counts_all = {arm: [] for arm in arms}

        for seed in SEEDS:
            run = all_data[f"{task_key}_adaptive"][seed]
            count_hist = run["controller_logs"]["lr_controller_logs"]["warmup_counts_history"]
            n_episodes = len(count_hist)

            for arm in arms:
                cum = np.array([entry[str(arm)] for entry in count_hist], dtype=float)
                total = np.zeros(n_episodes)
                for arm2 in arms:
                    total += np.array([entry[str(arm2)] for entry in count_hist], dtype=float)
                total = np.maximum(total, 1)
                frac = cum / total
                all_fracs[arm].append(frac)
                final_counts_all[arm].append(int(cum[-1]))

        for arm in arms:
            arr = np.array(all_fracs[arm])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1)
            episodes = np.arange(1, len(mean) + 1)
            ax.plot(episodes, mean, color=arm_colors[arm], label=arm_labels[arm], zorder=3)
            ax.fill_between(episodes, mean - std, mean + std, color=arm_colors[arm], alpha=0.15, zorder=2)

        ax.axhline(1 / 3, color="0.6", linewidth=0.7, linestyle=":", zorder=1, label="Uniform (1/3)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Selection Fraction" if col == 0 else "")
        ax.set_title(title)
        ax.legend(loc="right", fontsize=8.5)
        ax.set_xlim(1, len(mean))
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.grid(True)

        text_lines = []
        for arm in arms:
            mean_count = np.mean(final_counts_all[arm])
            pct = mean_count / sum(np.mean(final_counts_all[a]) for a in arms) * 100
            text_lines.append(f"$\\alpha$={arm}: {mean_count:.0f} pulls ({pct:.0f}%)")
        ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
        )

    fig.suptitle("Bandit Arm-Selection Frequency Over Training (mean ± 1 std, 5 seeds)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_controller_arm_selection")
    print("Fig 5 done")


def fig6_paired_ttest_summary(all_data):
    apply_thesis_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    tasks = ["CIFAR-100 (Clean)", "CIFAR-100 (Noisy)", "SST-2"]
    task_keys = ["cifar_clean", "cifar_noisy", "sst2"]
    bar_colors = ["#92c5de", "#4393c3", "#d1e5f0"]

    deltas_pp = []
    ci_pp = []
    p_values = []

    for task_key in task_keys:
        adaptive_vals = np.array(get_vals(all_data, f"{task_key}_adaptive", "best_val_accuracy"))
        fixed_vals = np.array(get_vals(all_data, f"{task_key}_fixed", "best_val_accuracy"))
        paired_deltas = (adaptive_vals - fixed_vals) * 100

        deltas_pp.append(np.mean(paired_deltas))
        ci_pp.append(1.96 * np.std(paired_deltas, ddof=1) / np.sqrt(len(paired_deltas)))
        _, p_val = stats.ttest_rel(adaptive_vals, fixed_vals)
        p_values.append(p_val)

    x = np.arange(len(tasks))
    bars = ax.bar(
        x,
        deltas_pp,
        yerr=ci_pp,
        width=0.58,
        color=bar_colors,
        edgecolor=C_ADAPTIVE,
        linewidth=1.1,
        capsize=4,
        error_kw={"linewidth": 1.0},
        zorder=3,
    )

    ax.axhline(0, color="0.35", linewidth=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Adaptive − Fixed Best Val Acc (pp)")
    ax.set_title("Paired t-test Summary Across Tasks")
    ax.yaxis.grid(True)

    def stars(p_val):
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    for i, (bar, delta, err, p_val) in enumerate(zip(bars, deltas_pp, ci_pp, p_values)):
        label_y = delta + err + 0.12 if delta >= 0 else delta - err - 0.32
        va = "bottom" if delta >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{delta:+.2f} pp\np={p_val:.3g} {stars(p_val)}",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold",
            color="#1a7a1a" if p_val < 0.05 else "#555555",
        )

    ymin = min(delta - err for delta, err in zip(deltas_pp, ci_pp)) - 0.5
    ymax = max(delta + err for delta, err in zip(deltas_pp, ci_pp)) + 0.6
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    save(fig, "fig6_paired_ttest_summary")
    print("Fig 6 done")


def main():
    all_data = load_all()
    plot2b_training_curves_sst2(all_data)
    plot3_bandit_dynamics(all_data)
    plot4_seed_paired_slopes(all_data)
    plot8_training_loss(all_data)
    fig1_summary_bar(all_data)
    fig2_best_vs_final(all_data)
    fig3_per_seed_delta(all_data)
    fig4_cifar_val_curves(all_data)
    fig5_controller_arm_selection(all_data)
    fig6_paired_ttest_summary(all_data)
    print(f"\nAll fixed-vs-adaptive plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
