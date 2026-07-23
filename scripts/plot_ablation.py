#!/usr/bin/env python3
"""
plot_ablation.py — Generate ablation study figures from run_ablation.sh output.

Usage:
    python scripts/plot_ablation.py --summary logs/ablation_<timestamp>/summary.csv
    python scripts/plot_ablation.py --summary logs/ablation_<timestamp>/summary.csv --output paper/figures/

Reads the summary.csv produced by run_ablation.sh and generates:
  1. Epsilon sensitivity line plot (Fig: epsilon_sensitivity.pdf)
  2. A/B gate bar chart (Fig: abgate_ablation.pdf)
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "mmlu": "#2563eb",
    "ifeval": "#dc2626",
}
MARKERS = {"mmlu": "o", "ifeval": "s"}


def parse_summary(path: str):
    """Parse summary.csv into structured dicts."""
    epsilon_data = defaultdict(dict)   # {benchmark: {eps_value: accuracy}}
    abgate_data = defaultdict(dict)    # {benchmark: {variant: accuracy}}

    eps_label_map = {
        "eps0": 0.0,
        "eps005": 0.005,
        "eps01": 0.01,
        "eps02": 0.02,
        "eps05": 0.05,
        "eps100": 1.0,
    }
    abgate_label_map = {
        "off": "Off\n(δ=−1)",
        "zero": "Neutral\n(δ=0)",
        "default": "Default\n(δ=0.02)",
        "strict": "Strict\n(δ=0.05)",
    }
    abgate_order = ["off", "zero", "default", "strict"]

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            try:
                acc = float(row["final_accuracy"])
            except (ValueError, KeyError):
                continue

            if label.startswith("epsilon_"):
                # e.g. epsilon_mmlu_eps01
                parts = label.split("_", 2)  # ["epsilon", "mmlu", "eps01"]
                bench = parts[1]
                eps_key = parts[2]
                if eps_key in eps_label_map:
                    epsilon_data[bench][eps_label_map[eps_key]] = acc

            elif label.startswith("abgate_"):
                # e.g. abgate_mmlu_default
                parts = label.split("_", 2)
                bench = parts[1]
                variant = parts[2]
                abgate_data[bench][variant] = acc

    return epsilon_data, abgate_data, abgate_label_map, abgate_order


def plot_epsilon_sensitivity(epsilon_data, output_dir):
    """Line plot: accuracy vs epsilon for each benchmark."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for bench in sorted(epsilon_data.keys()):
        eps_vals = sorted(epsilon_data[bench].keys())
        accs = [epsilon_data[bench][e] for e in eps_vals]
        # Use log scale for x but show actual values
        ax.plot(
            eps_vals, accs,
            marker=MARKERS.get(bench, "^"),
            color=COLORS.get(bench, "#666"),
            label=bench.upper(),
            linewidth=1.8,
            markersize=7,
        )
        # Highlight the default (0.01)
        if 0.01 in epsilon_data[bench]:
            ax.axvline(x=0.01, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Prompt Saturation Threshold (ε)")
    ax.set_ylabel("Test Accuracy")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([0.0, 0.005, 0.01, 0.02, 0.05, 1.0])
    ax.get_xaxis().set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Credit-Assignment Sensitivity (Qwen3-4B)")
    fig.tight_layout()

    path = os.path.join(output_dir, "epsilon_sensitivity.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_abgate_ablation(abgate_data, abgate_label_map, abgate_order, output_dir):
    """Grouped bar chart: accuracy per A/B gate variant for each benchmark."""
    benchmarks = sorted(abgate_data.keys())
    n_bench = len(benchmarks)
    n_variants = len(abgate_order)
    x = np.arange(n_variants)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for i, bench in enumerate(benchmarks):
        accs = [abgate_data[bench].get(v, 0.0) for v in abgate_order]
        offset = (i - (n_bench - 1) / 2) * width
        bars = ax.bar(
            x + offset, accs, width,
            label=bench.upper(),
            color=COLORS.get(bench, "#666"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{acc:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("A/B Gate Threshold")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([abgate_label_map[v] for v in abgate_order])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Structure Update A/B Gate Ablation (Qwen3-4B)")
    fig.tight_layout()

    path = os.path.join(output_dir, "abgate_ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--summary", required=True, help="Path to summary.csv from run_ablation.sh")
    parser.add_argument("--output", default="paper/figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    epsilon_data, abgate_data, abgate_label_map, abgate_order = parse_summary(args.summary)

    if epsilon_data:
        plot_epsilon_sensitivity(epsilon_data, args.output)
    else:
        print("No epsilon ablation data found in summary.")

    if abgate_data:
        plot_abgate_ablation(abgate_data, abgate_label_map, abgate_order, args.output)
    else:
        print("No A/B gate ablation data found in summary.")


if __name__ == "__main__":
    main()
