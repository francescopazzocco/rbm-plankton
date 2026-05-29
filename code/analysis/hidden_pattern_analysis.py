from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


DEFAULT_INPUT_DIR = Path("weights/NB_RBM/L6_chrono")
DEFAULT_OUTPUT_DIR = Path("analysis/results/hidden_patterns")


def load_hidden_activations(input_dir: Path) -> pd.DataFrame:
    csv_path = input_dir / "rbm_hidden_activations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing hidden-activations CSV: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    hidden_cols = [col for col in df.columns if col.startswith("h")]
    if not hidden_cols:
        raise ValueError(f"No hidden-unit columns found in {csv_path}")
    return df[["date", *hidden_cols]].sort_values("date").reset_index(drop=True)


def normalize_hidden_probs(hidden_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Convert hidden probabilities to a binary pattern representation.

    sigmoid: threshold each unit at 0.5.
    softmax: winner-take-all one-hot pattern.
    """
    values = hidden_df.to_numpy(dtype=float)
    n_rows, n_hidden = values.shape

    if mode == "sigmoid":
        binary = (values >= 0.5).astype(np.int8)
    elif mode == "softmax":
        binary = np.zeros_like(values, dtype=np.int8)
        winners = values.argmax(axis=1)
        binary[np.arange(n_rows), winners] = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pd.DataFrame(binary, columns=hidden_df.columns, index=hidden_df.index)


def pattern_labels(binary_df: pd.DataFrame) -> pd.Series:
    """Create binary-string labels like 010101 for each row."""
    labels = binary_df.astype(int).astype(str).agg("".join, axis=1)
    return labels.rename("pattern")


def pattern_summary(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hidden_cols = [col for col in df.columns if col.startswith("h")]
    binary_df = normalize_hidden_probs(df[hidden_cols], mode)
    patterns = pattern_labels(binary_df)

    summary = (
        patterns.value_counts()
        .rename_axis("pattern")
        .reset_index(name="frequency")
        .sort_values(["frequency", "pattern"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["fraction"] = summary["frequency"] / len(patterns)
    summary["count_rank"] = np.arange(1, len(summary) + 1)

    timeline = df[["date"]].copy()
    timeline["pattern"] = patterns.values
    return summary, timeline


def plot_pattern_histogram(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(summary)), 7))

    x = np.arange(len(summary))
    ax.bar(x, summary["frequency"].to_numpy(), color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["pattern"].tolist(), rotation=90, fontsize=8)
    ax.set_ylabel("Frequency")
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
        description="Analyze hidden-unit activation patterns from saved RBM hidden probabilities."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing rbm_hidden_activations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and tables are saved",
    )
    parser.add_argument(
        "--mode",
        choices=["sigmoid", "softmax"],
        default="sigmoid",
        help="How to convert hidden probabilities to a discrete pattern",
    )
    parser.add_argument(
        "--title-prefix",
        default="NB-RBM",
        help="Prefix used in figure titles",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = load_hidden_activations(args.input_dir)
    summary, timeline = pattern_summary(df, args.mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"pattern_frequency_{args.mode}.csv"
    timeline_path = args.output_dir / f"pattern_timeline_{args.mode}.csv"
    summary.to_csv(summary_path, index=False)
    timeline.to_csv(timeline_path, index=False)

    pattern_hist_path = args.output_dir / f"pattern_histogram_{args.mode}.png"
    pattern_timeline_path = args.output_dir / f"pattern_timeline_{args.mode}.png"

    plot_pattern_histogram(
        summary,
        pattern_hist_path,
        f"{args.title_prefix} hidden-pattern frequency ({args.mode})",
    )
    plot_pattern_timeline(
        timeline,
        pattern_timeline_path,
        f"{args.title_prefix} hidden-pattern timeline ({args.mode})",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved timeline: {timeline_path}")
    print(f"Saved histogram: {pattern_hist_path}")
    print(f"Saved timeline plot: {pattern_timeline_path}")


if __name__ == "__main__":
    main()