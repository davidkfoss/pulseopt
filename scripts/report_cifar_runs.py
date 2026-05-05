#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict


ROOT = Path("results")


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def safe_mean(xs):
    return mean(xs) if xs else None


def safe_std(xs):
    if not xs:
        return None
    return stdev(xs) if len(xs) > 1 else 0.0


def get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_run_name(data, path):
    cfg = data.get("config", {})
    label = cfg.get("experiment_label")
    if label:
        return label
    output = cfg.get("output") or cfg.get("output_dir")
    if output:
        return Path(output).name
    return path.parent.name


def load_rows(root: Path):
    rows = []
    for p in root.rglob("run_result.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        cfg = data.get("config", {})
        diagnostics = cfg.get("diagnostics", {})
        label_noise = diagnostics.get("label_noise", {})
        subset = diagnostics.get("final_subset_accuracies", {})

        row = {
            "path": str(p),
            "run_name": infer_run_name(data, p),
            "seed": data.get("seed", cfg.get("seed")),
            "optimizer": cfg.get("optimizer"),
            "method_name": data.get("method_name"),
            "control_mode": cfg.get("control_mode"),
            "label_noise_type": cfg.get("label_noise_type", "none"),
            "label_noise_rate": cfg.get("label_noise_rate", 0.0),
            "epochs": data.get("total_epochs", cfg.get("epochs")),
            "final_val_accuracy": data.get("final_val_accuracy"),
            "best_val_accuracy": data.get("best_val_accuracy"),
            "final_train_loss": data.get("final_train_loss"),
            "train_all_accuracy": subset.get("train_all_accuracy"),
            "train_clean_accuracy": subset.get("train_clean_accuracy"),
            "train_corrupted_accuracy_vs_noisy_labels": subset.get("train_corrupted_accuracy_vs_noisy_labels"),
            "train_corrupted_accuracy_vs_clean_labels": subset.get("train_corrupted_accuracy_vs_clean_labels"),
            "num_train_examples": label_noise.get("num_train_examples"),
            "num_clean_examples": label_noise.get("num_clean_examples"),
            "num_corrupted_examples": label_noise.get("num_corrupted_examples"),
        }
        rows.append(row)
    return rows


def print_group_report(rows):
    grouped = defaultdict(list)
    for r in rows:
        key = (
            r["optimizer"],
            r["method_name"],
            r["label_noise_type"],
            r["label_noise_rate"],
            r["epochs"],
            r["run_name"],
        )
        grouped[key].append(r)

    print("\n==============================")
    print("CIFAR RESULT DIAGNOSTICS REPORT")
    print(f"root = {ROOT}")
    print("==============================")

    for key in sorted(grouped):
        optimizer, method_name, noise_type, noise_rate, epochs, run_name = key
        group = sorted(grouped[key], key=lambda x: (
            x["seed"] is None, x["seed"]))

        finals = [r["final_val_accuracy"]
                  for r in group if r["final_val_accuracy"] is not None]
        bests = [r["best_val_accuracy"]
                 for r in group if r["best_val_accuracy"] is not None]
        train_all = [r["train_all_accuracy"]
                     for r in group if r["train_all_accuracy"] is not None]
        train_clean = [r["train_clean_accuracy"]
                       for r in group if r["train_clean_accuracy"] is not None]
        corr_noisy = [r["train_corrupted_accuracy_vs_noisy_labels"]
                      for r in group if r["train_corrupted_accuracy_vs_noisy_labels"] is not None]
        corr_clean = [r["train_corrupted_accuracy_vs_clean_labels"]
                      for r in group if r["train_corrupted_accuracy_vs_clean_labels"] is not None]

        print(f"\nRUN: {run_name}")
        print(
            f"  optimizer={optimizer} method={method_name} noise={noise_type} rate={noise_rate} epochs={epochs}")
        print(f"  n_seeds={len(group)}")
        print(
            f"  final_val_accuracy_mean={fmt(safe_mean(finals))} std={fmt(safe_std(finals))}")
        print(
            f"  best_val_accuracy_mean={fmt(safe_mean(bests))} std={fmt(safe_std(bests))}")
        print(
            f"  train_all_accuracy_mean={fmt(safe_mean(train_all))} std={fmt(safe_std(train_all))}")
        print(
            f"  train_clean_accuracy_mean={fmt(safe_mean(train_clean))} std={fmt(safe_std(train_clean))}")
        print(
            f"  corrupted_vs_noisy_mean={fmt(safe_mean(corr_noisy))} std={fmt(safe_std(corr_noisy))}")
        print(
            f"  corrupted_vs_clean_mean={fmt(safe_mean(corr_clean))} std={fmt(safe_std(corr_clean))}")

        if corr_noisy and corr_clean:
            deltas = [b - a for a, b in zip(corr_noisy, corr_clean)]
            print(
                f"  corrupted_clean_minus_noisy_mean={fmt(safe_mean(deltas))} std={fmt(safe_std(deltas))}")

        sample = group[0]
        if sample["num_train_examples"] is not None:
            print("  corruption_counts="
                  f"{sample['num_corrupted_examples']}/{sample['num_train_examples']} "
                  f"(clean={sample['num_clean_examples']})")

        print("  per-seed:")
        for r in group:
            print(
                f"    seed={r['seed']} "
                f"final_val={fmt(r['final_val_accuracy'])} "
                f"best_val={fmt(r['best_val_accuracy'])} "
                f"train_clean={fmt(r['train_clean_accuracy'])} "
                f"corr_noisy={fmt(r['train_corrupted_accuracy_vs_noisy_labels'])} "
                f"corr_clean={fmt(r['train_corrupted_accuracy_vs_clean_labels'])}"
            )


if __name__ == "__main__":
    rows = load_rows(ROOT)
    if not rows:
        print("No run_result.json files found.")
    else:
        print_group_report(rows)
