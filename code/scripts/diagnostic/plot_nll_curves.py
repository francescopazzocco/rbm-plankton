"""
plot_nll_curves.py - Train/val NLL training curves of several families at one L.

Mean ± 1σ over seeds per family, train and val side by side, so two
hidden-unit variants can be compared epoch by epoch rather than only at the
last epoch (sweep_analysis.py's final-metric plots). One family across several L: plot_train_nll_curves.py.

Output: diagnostic_outputs/diagnostics/training_curves/family_comparison_fixed_L/{split}/{family}_vs_..._L{L}.png

Usage:
    python code/scripts/diagnostic/plot_nll_curves.py \\
        [--families nb nb_sigmoid] [--L 6] [--split shuffled]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.io import SHUFFLED, SPLITS, model_dir, split_out_dir
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import COLORS, aggregate_curves, display_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/val NLL curves of several families at one L.")
    parser.add_argument("--families", nargs="+", default=["nb", "nb_sigmoid"],
                        help="NB/ZINB families to overlay (default: nb nb_sigmoid)")
    parser.add_argument("--L", type=int, default=6, help="Hidden unit count (default: 6)")
    parser.add_argument("--split", choices=SPLITS, default=SHUFFLED,
                        help="Split strategy (default: shuffled)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out_dir = split_out_dir(DIAGNOSTIC_ROOT / "diagnostics" / "training_curves" / "family_comparison_fixed_L", args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for family in args.families:
        csvs = sorted(model_dir(family, args.L, args.split, args.models_root)
                      .glob("seed_*/rbm_training_curves.csv"))
        for ax, col in zip(axes, ["train_nll", "val_nll"]):
            agg = aggregate_curves(csvs, col)
            if agg is None:
                continue
            mean, std = agg
            ax.plot(mean.index, mean.values, color=COLORS[family], lw=1.8,
                    label=f"{display_name(family)}  (final {mean.iloc[-1]:.4f} ± {std.iloc[-1]:.4f})")
            ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[family], alpha=0.2)

    for ax, title in zip(axes, ["Train NLL", "Val NLL"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("NLL per entry")
    fig.suptitle(f"{' vs '.join(display_name(f) for f in args.families)} - L={args.L} "
                 f"({args.split}), mean ± 1σ over seeds", fontweight="bold")
    fig.tight_layout()
    out = out_dir / f"{'_vs_'.join(args.families)}_L{args.L}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
