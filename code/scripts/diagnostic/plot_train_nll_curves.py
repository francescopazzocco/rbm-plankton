"""
plot_train_nll_curves.py - Train NLL vs epoch of one family across several L.

Mean ± 1σ over seeds, one line per L. Complements plot_nll_curves.py (several
families at one L) and replaces the archived plot_sigmoid_nll.py /
plot_zinb_nll.py, which read the pre-artifacts/ layout.

Output: diagnostic_outputs/diagnostics/training_curves/single_family_by_L/{split}/{family}_train_nll.png

Usage:
    python code/scripts/diagnostic/plot_train_nll_curves.py \\
        --families nb_softmax zinb_sigmoid zinb_softmax [--L 4 5 6 7 8] [--split shuffled]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.io import SHUFFLED, SPLITS, model_dir, split_out_dir
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import aggregate_curves, display_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NLL vs epoch of one family across several L.")
    parser.add_argument("--families", nargs="+", required=True, help="Families to plot, one figure each")
    parser.add_argument("--L", nargs="+", type=int, default=[4, 5, 6, 7, 8],
                        help="Hidden unit counts (default: 4 5 6 7 8)")
    parser.add_argument("--split", choices=SPLITS, default=SHUFFLED,
                        help="Split strategy (default: shuffled)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out_dir = split_out_dir(DIAGNOSTIC_ROOT / "diagnostics" / "training_curves" / "single_family_by_L",
                            args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.colormaps["viridis"]

    for family in args.families:
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, l_val in enumerate(args.L):
            csvs = sorted(model_dir(family, l_val, args.split, args.models_root)
                          .glob("seed_*/rbm_training_curves.csv"))
            agg = aggregate_curves(csvs, "train_nll")
            if agg is None:
                continue
            mean, std = agg
            color = cmap(i / max(len(args.L) - 1, 1))
            ax.plot(mean.index, mean.values, color=color, lw=1.5, label=f"L={l_val}")
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train NLL")
        ax.set_title(f"{display_name(family)} - Train NLL ({args.split}, mean ± 1σ over seeds)")
        ax.legend(title="Hidden units")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"{family}_train_nll.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
