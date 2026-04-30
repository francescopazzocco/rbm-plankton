"""
hidden_coactivation.py — Community state analysis via weight profiles and temporal assignment.

Two figures per model family:
  1. Weight profiles: species × hidden unit heatmap (species sorted by dominant unit)
     → shows what each hidden unit encodes ecologically
  2. Dominant state timeline: each date assigned to its argmax hidden unit, one row per L
     → shows whether the same temporal structure is recovered across L values

Output: results/hidden/weight_profiles_{family}.png
        results/hidden/state_timeline_{family}.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR     = Path(__file__).parent.parent / "results" / "hidden"

FAMILIES = ["bernoulli_median", "bernoulli_zero", "nb"]


def discover_runs(results_dir: Path) -> dict[str, dict[int, dict[str, Path]]]:
    """Return {family: {L: {activations, weights}}}."""
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, dict[str, Path]]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        act_csv = d / "rbm_hidden_activations.csv"
        w_csv   = d / "rbm_weights.csv"
        if act_csv.exists() and w_csv.exists():
            runs.setdefault(family, {})[l_val] = {
                "activations": act_csv,
                "weights":     w_csv,
            }
    return runs


# ── Figure 1: weight profiles ────────────────────────────────────────────────

def load_weights(csv: Path) -> pd.DataFrame:
    """Return DataFrame (species × hidden units)."""
    return pd.read_csv(csv, index_col=0)


TOP_SPECIES_PER_UNIT = 8  # top species by |weight| to include per hidden unit


def select_top_species(W: pd.DataFrame) -> pd.DataFrame:
    """Keep only the union of top-N species per hidden unit, sorted by dominant unit."""
    top_idx = set()
    for col in W.columns:
        top_idx.update(W[col].abs().nlargest(TOP_SPECIES_PER_UNIT).index)
    W_filtered = W.loc[sorted(top_idx)]
    dominant = W_filtered.abs().values.argmax(axis=1)
    order = np.argsort(dominant, kind="stable")
    return W_filtered.iloc[order]


def plot_weight_profiles(family: str, family_runs: dict[int, dict], out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)

    fig, axes = plt.subplots(1, n_l, figsize=(4 * n_l, 7), sharey=False)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(
        f"{family} — weight profiles (top-{TOP_SPECIES_PER_UNIT} species per unit)",
        fontsize=11
    )

    for ax, l_val in zip(axes, l_values):
        W = load_weights(family_runs[l_val]["weights"])
        W_top = select_top_species(W)

        vmax = np.abs(W_top.values).max()
        im = ax.imshow(W_top.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax.set_title(f"L={l_val}  ({len(W_top)} species shown)", fontsize=9)
        ax.set_xlabel("hidden unit", fontsize=8)
        ax.set_xticks(range(l_val))
        ax.set_xticklabels([f"h{i}" for i in range(l_val)], fontsize=8)
        ax.set_yticks(range(len(W_top)))
        ax.set_yticklabels(W_top.index, fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="weight")

    fig.tight_layout()
    out = out_dir / f"weight_profiles_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 2: dominant state timeline ────────────────────────────────────────

def load_activations(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, index_col="date", parse_dates=True)
    return df


def dominant_state(activations: pd.DataFrame) -> pd.Series:
    """Assign each sample to the hidden unit with highest activation."""
    return pd.Series(
        activations.values.argmax(axis=1),
        index=activations.index,
        name="dominant"
    )


def plot_state_timeline(family: str, family_runs: dict[int, dict], out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)

    fig, axes = plt.subplots(n_l, 1, figsize=(14, 2.2 * n_l), sharex=True)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(f"{family} — dominant hidden state over time", fontsize=11)

    # Build a consistent colormap across all L (up to max L units)
    max_l = max(l_values)
    cmap  = plt.colormaps["tab10"]

    for ax, l_val in zip(axes, l_values):
        act   = load_activations(family_runs[l_val]["activations"])
        state = dominant_state(act)
        dates = state.index

        colors = [cmap(s / max_l) for s in state.values]
        ax.scatter(dates, np.zeros(len(dates)), c=colors,
                   marker="|", s=200, linewidths=2)
        ax.set_yticks([])
        ax.set_ylabel(f"L={l_val}", rotation=0, labelpad=30, fontsize=9, va="center")
        ax.set_xlim(dates.min(), dates.max())

        # Legend for this L
        handles = [
            plt.Line2D([0], [0], marker="|", color="w",
                       markerfacecolor=cmap(i / max_l),
                       markeredgecolor=cmap(i / max_l),
                       markersize=10, label=f"h{i}")
            for i in range(l_val)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=l_val, framealpha=0.7)

    axes[-1].set_xlabel("date", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"state_timeline_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    for family in FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        print(f"\n── {family} ──")
        plot_weight_profiles(family, runs[family], OUT_DIR)
        plot_state_timeline(family, runs[family], OUT_DIR)


if __name__ == "__main__":
    main()
