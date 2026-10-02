"""plot_sample_reconstruction.py - One validation day: ground truth, reconstruction, residuals.

Draws one random day from the held-out rows of a shuffled run and reconstructs
it with each requested model (one visible -> hidden -> visible pass, reading
the NB mean of the visible layer). Top panel: abundance per taxon, ground truth
(line) against the reconstructions (bars); bottom panel: residual (reconstruction - truth).

Bernoulli hidden units are sampled, so their reconstruction is averaged over
--draws hidden samples (black ticks: 5-95% range of those draws); Sigmoid
hidden units pass their probabilities and need no averaging.

All models must share the seed: the shuffled split is drawn from the run's
seed, so only runs with the same seed hold out the same days.

Usage:
    python code/scripts/analysis/reconstruction/plot_sample_reconstruction.py \
        [--families nb nb_sigmoid] [--L 6] [--seed 9] [--row-seed 0]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from use_trained_rbm import (  # noqa: E402
    instantiate_model_from_weights,
    load_evaluation_split,
    load_weights_npz,
)

from models import io as data_io  # noqa: E402
from models.paths import DIAGNOSTIC_ROOT, SHUFFLED, model_dir  # noqa: E402
from models.plot_style import FIG_DPI, apply_style  # noqa: E402
from models.visualization import COLORS, display_name  # noqa: E402

apply_style()


@torch.no_grad()
def reconstruct_mean(model, v: torch.Tensor, draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean reconstruction of one row and the 5-95% range over hidden draws."""
    V = v.unsqueeze(0).repeat(draws, 1)
    rec = model.reconstruct(V).cpu().numpy()
    return rec.mean(0), np.percentile(rec, 5, axis=0), np.percentile(rec, 95, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--families", nargs="+", default=["nb", "nb_sigmoid"])
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--seed", type=int, default=9, help="Training seed shared by every model")
    parser.add_argument("--row-seed", type=int, default=0, help="RNG seed for picking the validation day")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--out", type=Path,
                        default=DIAGNOSTIC_ROOT / "reconstruction_plots" / "sample_day")
    args = parser.parse_args()

    device = torch.device("cpu")
    X_val, taxa = load_evaluation_split("nb", device, shuffle=True, seed=args.seed)
    # dates of the same held-out rows, drawn with the same seed
    np.random.seed(args.seed)
    *_, dates_val, _, _ = data_io.load_raw_counts(device=device, shuffle=True)

    row = int(np.random.default_rng(args.row_seed).integers(X_val.shape[0]))
    v = X_val[row]
    truth = v.cpu().numpy()
    day = str(np.asarray(dates_val)[row])[:10]

    recs = {}
    for fam in args.families:
        w = model_dir(fam, args.L, SHUFFLED) / f"seed_{args.seed}" / "weights.npz"
        model, _ = instantiate_model_from_weights(load_weights_npz(w), device_str="cpu")
        recs[fam] = reconstruct_mean(model, v.to(model.device), args.draws)

    x = np.arange(len(taxa))
    width = 0.8 / len(recs)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(18, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for k, (fam, (mean, lo, hi)) in enumerate(recs.items()):
        xs = x - 0.4 + width / 2 + k * width
        l1 = np.abs(mean - truth).mean()
        ax.bar(xs, mean, width, color=COLORS[fam],
               label=f"{display_name(fam)} L={args.L}  (mean L1 = {l1:.3f})")
        if not np.allclose(lo, hi):
            ax.vlines(xs, lo, hi, color="black", lw=0.6)
        axr.bar(xs, mean - truth, width, color=COLORS[fam])
    # ground truth as a line, so the bars only carry the reconstructions and stay wide
    ax.plot(x, truth, color="black", lw=1.4, marker="o", ms=3, label="ground truth", zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("abundance (counts/µL × 1000)")
    ax.legend(loc="upper right", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    axr.axhline(0, color="black", lw=0.8)
    axr.set_yscale("symlog", linthresh=1.0)
    axr.set_ylabel("residual\n(reconstruction − truth)")
    axr.grid(axis="y", alpha=0.3)
    axr.set_xticks(x, [t.replace("_", " ") for t in taxa], rotation=45, ha="right",
                   rotation_mode="anchor", fontsize=8)
    axr.set_xlim(-0.6, len(taxa) - 0.4)
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"sample_{day}_L{args.L}_seed{args.seed}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    for fam, (mean, _, _) in recs.items():
        print(f"{display_name(fam)}: mean L1 = {np.abs(mean - truth).mean():.4f}")
    print(f"day {day} (row {row} of {X_val.shape[0]}), saved: {out}")


if __name__ == "__main__":
    main()
