"""
hidden_mean_activation.py — Mean hidden unit activation across L values.

For each model family and L, computes the mean activation of each hidden unit
over all samples. Identifies always-on (bias absorber) and always-off units.

Output: results/hidden/mean_activation_{family}.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_DIR     = Path(__file__).parent.parent / "results" / "hidden"

FAMILIES = ["bernoulli_median", "bernoulli_zero", "nb"]

# Units with mean activation above/below these thresholds are flagged
ABSORBER_HI = 0.90
ABSORBER_LO = 0.10


def discover_runs(results_dir: Path) -> dict[str, dict[int, Path]]:
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, Path]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        csv = d / "rbm_hidden_activations.csv"
        if csv.exists():
            runs.setdefault(family, {})[l_val] = csv
    return runs


def mean_activations(csv: Path) -> pd.Series:
    df = pd.read_csv(csv, index_col="date")
    return df.mean()


def plot_family(family: str, family_runs: dict[int, Path], out_dir: Path):
    l_values = sorted(family_runs)
    n = len(l_values)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{family} — mean hidden activation per unit", fontsize=12)

    for ax, l_val in zip(axes, l_values):
        means = mean_activations(family_runs[l_val])
        units = np.arange(len(means))
        colors = [
            "#d62728" if v >= ABSORBER_HI else
            "#7f7f7f" if v <= ABSORBER_LO else
            "#1f77b4"
            for v in means.values
        ]
        ax.bar(units, means.values, color=colors)
        ax.axhline(ABSORBER_HI, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(ABSORBER_LO, color="#7f7f7f", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"L={l_val}", fontsize=10)
        ax.set_xlabel("hidden unit")
        ax.set_xticks(units)
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("mean p(h=1)")

        for i, v in enumerate(means.values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#d62728", label=f"always-on  (>{ABSORBER_HI})"),
        Patch(color="#7f7f7f", label=f"always-off (<{ABSORBER_LO})"),
        Patch(color="#1f77b4", label="active"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    out = out_dir / f"mean_activation_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    for family in FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        plot_family(family, runs[family], OUT_DIR)


if __name__ == "__main__":
    main()
