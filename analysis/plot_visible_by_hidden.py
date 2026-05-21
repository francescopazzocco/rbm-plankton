#!/usr/bin/env python3
"""Plot visible-unit probabilities when activating hidden nodes or patterns.

Loads weights from .npz (keys: W, a, b, log_theta, logit_pi) or from
rbm_weights.csv (D x L matrix, index=taxa names). Produces scatter plots
with species on x-axis and probability/mean on y-axis. Uses `rainbow`
colormap. Saves CSVs with computed probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_weights(path: Path):
    """Return (W, a, b, extras dict, taxa_names list-or-none).

    W shape: (D, L)
    a shape: (D,)
    b shape: (L,)
    extras: may contain 'log_theta', 'logit_pi'
    """
    path = Path(path)
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=True)
        W = npz["W"]
        a = npz["a"] if "a" in npz else np.zeros(W.shape[0])
        b = npz["b"] if "b" in npz else np.zeros(W.shape[1])
        extras = {k: npz[k] for k in ["log_theta", "logit_pi"] if k in npz}
        taxa = list(npz["taxa"]) if "taxa" in npz else None
        return W, a, b, extras, taxa

    # assume CSV with taxa index (D x L)
    df = pd.read_csv(path, index_col=0)
    W = df.values.astype(float)
    taxa = df.index.tolist()
    a = np.zeros(W.shape[0])
    b = np.zeros(W.shape[1])
    extras = {}
    return W, a, b, extras, taxa


def compute_visible_probs_for_hidden(W, a, b, extras, mode: str, H_vec: np.ndarray) -> np.ndarray:
    """Compute visible probabilities/means given hidden activation vector H_vec.

    mode: 'bernoulli' -> sigmoid(a + W @ H)
          'zinb'     -> (1-pi) * exp(a + W @ H)  (expected count/mean)
    """
    z = a + W.dot(H_vec)
    if mode == "bernoulli":
        return sigmoid(z)
    elif mode == "zinb":
        # safe exp clamp
        zc = np.clip(z, None, 10.0)
        mu = np.exp(zc)
        if "logit_pi" in extras:
            pi = sigmoid(extras["logit_pi"])
            return (1.0 - pi) * mu
        return mu
    else:
        raise ValueError("Unknown mode")


def plot_scatter(species, probs_list, labels, out_path: Path, title: str):
    """Legacy multi-series scatter. Keeps behaviour for small L."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D = len(species)
    N = len(probs_list)
    fig, ax = plt.subplots(figsize=(max(10, 0.4 * D), 6))

    cmap = matplotlib.colormaps["rainbow"]
    colors = cmap(np.linspace(0, 1, N, endpoint=False))

    # x base positions
    x = np.arange(D)
    # offsets to separate points for multiple groups
    if N > 1:
        offsets = (np.arange(N) - (N - 1) / 2.0) * (0.8 / (N - 1))
    else:
        offsets = np.array([0.0])

    for i, probs in enumerate(probs_list):
        xs = x + offsets[i]
        ax.scatter(
            xs,
            probs,
            facecolors=colors[i],
            marker="o",
            s=30,
            alpha=0.85,
            label=labels[i],
            edgecolors="black",
            linewidths=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=90, fontsize=8)
    ax.set_xlabel("Species")
    ax.set_ylabel("Probability / expected count")
    ax.set_title(title)
    if D > 1:
        ax.set_xlim(-0.05, D - 1 + 0.05)
    ymax = float(np.max([np.max(p) for p in probs_list])) if probs_list else 1.0
    ax.set_ylim(0.0, max(1e-9, 1.1 * ymax))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_per_hidden_rows(species, probs_list, labels, out_path: Path, title: str, normalize: bool = True, log_y: bool = False, mode: str = "bernoulli"):
    """Plot one subplot per hidden node (rows).

    By default renormalize each hidden node to frequency [0,1]. Set `normalize=False`
    to plot raw probabilities/means. Set `log_y=True` to use log scale (adds small
    epsilon to avoid zeros).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    D = len(species)
    L = len(probs_list)
    fig, axes = plt.subplots(L, 1, figsize=(max(10, 0.4 * D), 2.5 * L), sharex=True)
    if L == 1:
        axes = [axes]

    cmap = matplotlib.colormaps["rainbow"]
    colors = cmap(np.linspace(0, 1, L, endpoint=False))

    x = np.arange(D)
    for j, ax in enumerate(axes):
        probs = probs_list[j]
        if normalize:
            s = probs.sum()
            vals = probs / s if s > 0 else np.zeros_like(probs)
            ax.set_ylim(0.0, 1.0)
        else:
            vals = probs
            # set y-limits relative to data to avoid excessive white space
            vmax = float(np.nanmax(vals)) if vals.size > 0 else 1.0
            vmin = float(np.nanmin(vals)) if vals.size > 0 else 0.0
            # expand a little for visibility
            lower = min(0.0, vmin - 0.05 * max(1.0, abs(vmin)))
            upper = max(vmax * 1.05, 1e-9)
            ax.set_ylim(lower, upper)

        # force filled circle markers so edgecolors render reliably
        ax.scatter(x, vals, color=colors[j], s=30, alpha=0.9, marker="o", edgecolors="black", linewidths=0.35, zorder=3)
        ax.set_ylabel(labels[j], fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)

        if log_y:
            # use smallest positive value as floor, not a linear ylim in log space
            positive = vals[vals > 0]
            floor = float(np.min(positive)) if positive.size > 0 else 1e-6
            ceiling = float(np.max(vals)) if vals.size > 0 else floor * 10.0
            lower = max(floor / 1.5, 1e-9)
            upper = max(ceiling * 1.15, lower * 10.0)
            ax.set_yscale("log")
            ax.set_ylim(lower, upper)
            ax.axhline(floor, color="0.4", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(species, rotation=90, fontsize=8)
    axes[-1].set_xlabel("Species")
    # common ylabel describing what's plotted
    if normalize:
        ylab = "Frequency"
    else:
        if mode == "bernoulli":
            ylab = "Probability"
        elif mode == "zinb":
            ylab = "Expected count"
        else:
            ylab = "Value"
    if log_y:
        ylab = f"{ylab} (log scale)"
    fig.text(0.02, 0.5, ylab, va="center", rotation="vertical", fontsize=10)
    # reduce left/right padding so first/last ticks sit closer to axes
    if D > 1:
        axes[-1].set_xlim(-0.5, D - 0.5)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.18)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to weights .npz or rbm_weights.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results/visible_by_hidden"))
    parser.add_argument("--mode", choices=["bernoulli", "zinb"], default=None,
                        help="Interpretation for visible units. Auto-detect if omitted.")
    parser.add_argument("--patterns-csv", type=Path, default=None,
                        help="Optional CSV with column 'pattern' containing binary strings for hidden patterns")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Do not renormalize per-hidden to frequency [0,1]; plot raw values")
    parser.add_argument("--log-y", action="store_true",
                        help="Use log scale for y-axis (avoid zeros automatically)")
    parser.add_argument("--title-prefix", default="RBM",
                        help="Title prefix for plots")
    args = parser.parse_args()

    W, a, b, extras, taxa = load_weights(args.weights)
    D, L = W.shape

    mode = args.mode
    if mode is None:
        mode = "zinb" if "logit_pi" in extras or "log_theta" in extras else "bernoulli"

    species = taxa if taxa is not None else [f"s{i}" for i in range(D)]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- single-hidden-node activations ---
    probs_list = []
    labels = []
    for j in range(L):
        H = np.zeros(L, dtype=float)
        H[j] = 1.0
        probs = compute_visible_probs_for_hidden(W, a, b, extras, mode, H)
        probs_list.append(probs)
        labels.append(f"h{j}")

    plot_path = out_dir / f"visible_by_hidden_{mode}.png"
    # per-hidden-row plot (renormalizes each hidden node to frequency [0,1])
    plot_per_hidden_rows(species, probs_list, labels, plot_path, f"{args.title_prefix} visible probs by hidden ({mode})",
                         normalize=(not args.no_normalize), log_y=args.log_y, mode=mode)

    # save raw CSV and normalized-frequency CSV
    df = pd.DataFrame({label: probs for label, probs in zip(labels, probs_list)}, index=species)
    csv_path = out_dir / f"visible_by_hidden_{mode}.csv"
    df.to_csv(csv_path)

    df_freq = pd.DataFrame({label: (probs / (probs.sum() if probs.sum() > 0 else 1.0))
                             for label, probs in zip(labels, probs_list)}, index=species)
    csv_path_freq = out_dir / f"visible_by_hidden_{mode}_freq.csv"
    df_freq.to_csv(csv_path_freq)

    print(f"Saved plot {plot_path}")
    print(f"Saved CSV {csv_path}")
    print(f"Saved normalized CSV {csv_path_freq}")

    # --- patterns mode if provided ---
    if args.patterns_csv is not None and Path(args.patterns_csv).exists():
        p_df = pd.read_csv(args.patterns_csv)
        if "pattern" not in p_df.columns:
            raise SystemExit("patterns CSV must contain 'pattern' column with binary strings")
        pat_list = p_df["pattern"].astype(str).tolist()
        probs_patterns = []
        labels_pat = []
        for k, pat in enumerate(pat_list):
            bits = np.array([int(c) for c in pat.strip()], dtype=float)
            if bits.size != L:
                raise SystemExit(f"Pattern length {bits.size} != hidden L {L}")
            probs = compute_visible_probs_for_hidden(W, a, b, extras, mode, bits)
            probs_patterns.append(probs)
            labels_pat.append(f"p{k}")

        plot_path2 = out_dir / f"visible_by_patterns_{mode}.png"
        plot_per_hidden_rows(species, probs_patterns, labels_pat, plot_path2,
             f"{args.title_prefix} visible by hidden patterns ({mode})",
             normalize=(not args.no_normalize), log_y=args.log_y, mode=mode)

        df2 = pd.DataFrame({label: probs for label, probs in zip(labels_pat, probs_patterns)}, index=species)
        csv_path2 = out_dir / f"visible_by_patterns_{mode}.csv"
        df2.to_csv(csv_path2)

        df2_freq = pd.DataFrame({label: (probs / (probs.sum() if probs.sum() > 0 else 1.0))
                     for label, probs in zip(labels_pat, probs_patterns)}, index=species)
        csv_path2_freq = out_dir / f"visible_by_patterns_{mode}_freq.csv"
        df2_freq.to_csv(csv_path2_freq)

        print(f"Saved plot {plot_path2}")
        print(f"Saved CSV {csv_path2}")
        print(f"Saved normalized CSV {csv_path2_freq}")


if __name__ == "__main__":
    main()
