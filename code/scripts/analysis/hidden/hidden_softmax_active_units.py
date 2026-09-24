"""
hidden_softmax_active_units.py - Number of hidden units a Softmax RBM actually
uses, as a function of L, across every seed.

A Softmax hidden layer assigns each day a share over L units (shares sum to 1),
so a unit that the model never uses sits near zero mean activation. A unit is
counted as active when its mean activation over all clean days is above
absorber_lo(family) (0.05 for Softmax). If the count stops growing with L,
the extra capacity is left unused: an estimate of the number of distinct
community states the data supports.

Output: results/02_model_analysis/hidden/softmax_active_units/{chrono,shuffled}/active_units.png
        results/02_model_analysis/hidden/softmax_active_units/{chrono,shuffled}/active_units.csv

Usage:
    python code/scripts/analysis/hidden/hidden_softmax_active_units.py [--split chrono|shuffled]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.io import CHRONO, SPLITS, discover_model_dirs, split_out_dir
from models.palette import OKABE_ITO
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import absorber_lo, display_name, mean_activations

FAMILIES = ["nb_softmax", "zinb_softmax"]
COLORS = {"nb_softmax": OKABE_ITO[5], "zinb_softmax": OKABE_ITO[6]}
X_OFFSET = {"nb_softmax": -0.08, "zinb_softmax": 0.08}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active Softmax hidden units vs L, all seeds.")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Which split strategy's runs to analyse (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="Directory holding artifacts/models/{family}/{split}/L{n} directories")
    return parser.parse_args(argv)


def count_active(all_dirs: dict) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        lo = absorber_lo(family)
        for l_val, seed_dirs in sorted(all_dirs.get(family, {}).items()):
            for seed_dir in seed_dirs:
                csv = seed_dir / "rbm_hidden_activations.csv"
                if not csv.exists():
                    continue
                means = mean_activations(csv)
                if not np.isfinite(means.values).all():
                    continue
                rows.append({"family": family, "L": l_val, "seed": seed_dir.name,
                             "n_active": int((means > lo).sum())})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    l_all = sorted(df["L"].unique())
    ax.plot(l_all, l_all, color="#9AA1AB", linestyle="--", linewidth=1,
            label="every unit used (y = L)")
    for family in FAMILIES:
        g = df[df["family"] == family].groupby("L")["n_active"]
        if g.ngroups == 0:
            continue
        stats = g.agg(["mean", "std", "count"]).reset_index()
        x = stats["L"] + X_OFFSET[family]
        ax.errorbar(x, stats["mean"], yerr=stats["std"], color=COLORS[family],
                    marker="o", markersize=7, linewidth=2, capsize=4,
                    label=f"{display_name(family)} (mean ± 1σ, {int(stats['count'].min())} seeds)")
    ax.set_xlabel("hidden units available, L")
    ax.set_ylabel(f"active units (mean activation > {absorber_lo(FAMILIES[0])})")
    ax.set_xticks(l_all)
    ax.set_ylim(0, max(l_all) + 0.5)
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = split_out_dir(
        DIAGNOSTIC_ROOT / "02_model_analysis" / "hidden" / "softmax_active_units", args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = count_active(discover_model_dirs(args.models_root, args.split))
    if df.empty:
        print("No Softmax runs found.")
        return
    df.to_csv(out_dir / "active_units.csv", index=False)
    print(df.groupby(["family", "L"])["n_active"].agg(["mean", "std", "count"]).round(2))
    plot(df, out_dir / "active_units.png")


if __name__ == "__main__":
    main()
