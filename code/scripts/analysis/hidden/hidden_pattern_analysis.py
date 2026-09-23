"""
hidden_pattern_analysis.py - Hidden-unit activation patterns of a single run.

Discretises the hidden activations of one (family, L, split) run into binary
patterns and reports which patterns the model actually uses, and when.

The discretisation itself lives in models.visualization (hidden_binary,
pattern_frequency) and is shared with hidden_cross_model.py, so the threshold
rule exists in exactly one place.

Outputs (results/02_model_analysis/hidden/patterns/{chrono,shuffled}/ by default):
  pattern_frequency_{family}_L{L}_{mode}.csv   pattern -> n_days, fraction, n_units_on
  pattern_timeline_{family}_L{L}_{mode}.csv    date -> pattern
  pattern_histogram_{family}_L{L}_{mode}.png
  pattern_timeline_{family}_L{L}_{mode}.png

Usage:
    python code/scripts/analysis/hidden/hidden_pattern_analysis.py
    python code/scripts/analysis/hidden/hidden_pattern_analysis.py --family nb_softmax --L 7 \
        --split shuffled --mode winner
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.io import (
    CHRONO,
    METRIC_COL,
    SPLITS,
    best_seed_dir,
    load_hidden_activations,
    model_dir,
    split_out_dir,
)
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import hidden_binary, pattern_frequency, pattern_labels
from scipy.cluster.hierarchy import leaves_list, linkage


def resolve_seed_dir(family: str, n_hidden: int, split: str, models_root: Path) -> Path:
    """Best-converged seed directory of one run, by the family's val metric."""
    family_l_dir = model_dir(family, n_hidden, split, models_root)
    seed_dir = best_seed_dir(family_l_dir, METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(
            f"No converged seed for {family} L={n_hidden} ({split}) in {family_l_dir}")
    return seed_dir


def plot_pattern_histogram(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(summary)), 7))

    x = np.arange(len(summary))
    ax.bar(x, summary["n_days"].to_numpy(), color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["pattern"].tolist(), rotation=90, fontsize=8)
    ax.set_ylabel("Days")
    ax.set_xlabel("Hidden pattern (binary string)")
    ax.set_title(title)
    ax.text(
        0.98,
        0.98,
        f"Identified patterns: {len(summary)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=3.0),
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.text(
        0.5,
        -0.01,
        "Binary label order: leftmost digit = h0 (MSB in the label), rightmost digit = last hidden unit (LSB).",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pattern_timeline(timeline: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_patterns = timeline["pattern"].drop_duplicates().tolist()
    if len(unique_patterns) > 1:
        pattern_bits = pd.DataFrame(
            [[int(bit) for bit in pattern] for pattern in unique_patterns],
            index=unique_patterns,
        )
        order = leaves_list(linkage(pattern_bits.values, method="average", metric="hamming"))
        unique_patterns = [unique_patterns[idx] for idx in order]

    pattern_to_y = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
    pattern_to_color = {}
    if unique_patterns:
        colors = plt.colormaps["rainbow"](np.linspace(0, 1, len(unique_patterns), endpoint=False))
        pattern_to_color = {pattern: color for pattern, color in zip(unique_patterns, colors, strict=False)}

    timeline = timeline.copy()
    timeline["y"] = timeline["pattern"].map(pattern_to_y)
    timeline["color"] = timeline["pattern"].map(pattern_to_color)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(timeline["date"], timeline["y"], s=16, c=timeline["color"].tolist(), alpha=0.75, linewidths=0)

    ax.set_yticks(list(pattern_to_y.values()))
    ax.set_yticklabels(list(pattern_to_y.keys()), fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Hidden pattern")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="x", alpha=0.2)
    ax.grid(True, axis="y", alpha=0.12)
    fig.text(
        0.5,
        -0.01,
        "Binary label order: leftmost digit = h0 (MSB in the label), rightmost digit = last hidden unit (LSB).",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse hidden-unit activation patterns of one trained run."
    )
    parser.add_argument("--family", default="nb",
                        help="Model family (default: nb)")
    parser.add_argument("--L", type=int, default=6,
                        help="Hidden unit count (default: 6)")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Split strategy of the run (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="Directory holding artifacts/models/{family}/{split}/L{n} directories")
    parser.add_argument("--seed-dir", type=Path, default=None,
                        help="Use this seed directory directly, ignoring "
                             "--family/--L/--split")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory where plots and tables are saved "
                             "(default: diagnostic_outputs/02_model_analysis/hidden/patterns/"
                             "{chrono,shuffled}/, split-aware)")
    parser.add_argument("--mode", choices=["threshold", "winner"], default="threshold",
                        help="threshold: each unit on above 0.5 (Bernoulli/sigmoid units); "
                             "winner: one-hot argmax (softmax units)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    seed_dir = args.seed_dir or resolve_seed_dir(
        args.family, args.L, args.split, args.models_root)
    title_prefix = f"{args.family} L={args.L} ({args.split})" \
        if args.seed_dir is None else str(seed_dir)
    print(f"Reading {seed_dir}")

    activations = load_hidden_activations(seed_dir / "rbm_hidden_activations.csv")
    binary = hidden_binary(activations, mode=args.mode)

    summary = pattern_frequency(binary)
    timeline = pattern_labels(binary).reset_index()

    output_dir = args.output_dir or split_out_dir(
        DIAGNOSTIC_ROOT / "02_model_analysis" / "hidden" / "patterns", args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.family}_L{args.L}_{args.mode}"
    summary_path = output_dir / f"pattern_frequency_{tag}.csv"
    timeline_path = output_dir / f"pattern_timeline_{tag}.csv"
    summary.to_csv(summary_path, index=False)
    timeline.to_csv(timeline_path, index=False)

    pattern_hist_path = output_dir / f"pattern_histogram_{tag}.png"
    pattern_timeline_path = output_dir / f"pattern_timeline_{tag}.png"

    plot_pattern_histogram(
        summary,
        pattern_hist_path,
        f"{title_prefix} hidden-pattern frequency ({args.mode})",
    )
    plot_pattern_timeline(
        timeline,
        pattern_timeline_path,
        f"{title_prefix} hidden-pattern timeline ({args.mode})",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved timeline: {timeline_path}")
    print(f"Saved histogram: {pattern_hist_path}")
    print(f"Saved timeline plot: {pattern_timeline_path}")


if __name__ == "__main__":
    main()
