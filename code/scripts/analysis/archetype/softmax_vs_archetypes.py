"""softmax_vs_archetypes.py - One-hot Softmax states against the dominant archetype, day by day.

The Softmax hidden layer assigns every day to exactly one unit; archetypal
analysis (data/archetypes/archetypes_k5_timeseries.csv) gives every day a weight per
archetype, which we reduce to its dominant archetype. Both are a hard
partition of the same days, so they can be compared directly, without going
through weight profiles:

  1. contingency heatmap: rows = archetypes, columns = Softmax units, cell =
     share of the archetype's days assigned to that unit (count annotated);
     units are ordered by the archetype they share most days with;
  2. two timeline strips, dominant archetype above and Softmax state below;
     each unit is coloured as the archetype it is matched to, the archetype
     whose days it captures the largest share of (so A5, which never holds a
     majority inside its unit, still gets a colour on the Softmax strip).

The figures use the best seed at --L. Because that is one seed and one L,
--sweep-L also scores every seed at each listed L and writes the mean and
spread of NMI and purity, showing whether the agreement depends on the choice.

Agreement is summarised by normalised mutual information (NMI, 0 = independent
partitions, 1 = identical up to relabelling) and by purity (share of days whose
unit's majority archetype equals their own archetype).

Usage:
    python code/scripts/analysis/archetype/softmax_vs_archetypes.py [--family nb_softmax] [--L 6] \
        [--sweep-L 4 5 6 7 8 9 10]
"""

import argparse
import glob
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from models.io import METRIC_COL, SHUFFLED, SPLITS, best_seed_dir, model_dir, split_out_dir
from models.palette import ARCHETYPE_COLORS
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.paths import PROJECT_ROOT as ROOT
from models.plot_style import FIG_DPI, apply_style, show_titles
from models.visualization import display_name, dominant_state, load_activations

apply_style()

ARCH_COLORS = ARCHETYPE_COLORS  # same as the Venn diagram and the archetype figures


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic-mean normalisation)."""
    joint = pd.crosstab(a, b).values.astype(float)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb))


def day_labels(seed_dir: Path, arch: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Days shared by one run and the archetypes, with Softmax state and dominant archetype."""
    states = dominant_state(load_activations(seed_dir / "rbm_hidden_activations.csv"))
    states.index = pd.to_datetime(states.index)
    days = states.index.intersection(arch.index).sort_values()
    return days, states.loc[days].values, arch.loc[days].values


def agreement(a: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """NMI and purity of the Softmax partition against the archetype partition."""
    ct = pd.crosstab(a, s).values
    return nmi(a, s), ct.max(axis=0).sum() / ct.sum()


def seed_sweep(family: str, ls: list[int], split: str, models_root: Path, arch: pd.Series) -> pd.DataFrame:
    """Agreement of every seed at every L, to show how much the figure's choice matters."""
    rows = []
    for n_hidden in ls:
        root = model_dir(family, n_hidden, split, models_root)
        best = best_seed_dir(root, METRIC_COL[family])
        for d in sorted(glob.glob(str(root / "seed_*"))):
            _, s, a = day_labels(Path(d), arch)
            score_nmi, purity = agreement(a, s)
            rows.append({"L": n_hidden, "seed": Path(d).name, "best": Path(d) == best,
                         "units_with_days": len(set(s)), "nmi": score_nmi, "purity": purity})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="nb_softmax")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--split", choices=SPLITS, default=SHUFFLED)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--archetypes", type=Path, default=ROOT / "data" / "archetypes" / "archetypes_k5_timeseries.csv")
    parser.add_argument("--sweep-L", type=int, nargs="*", default=[],
                        help="Also score every seed at these L (CSV + summary)")
    args = parser.parse_args()

    arch_w = pd.read_csv(args.archetypes, parse_dates=["date"]).set_index("date")
    arch_names = list(arch_w.columns)
    arch = pd.Series(arch_w.values.argmax(axis=1), index=arch_w.index, name="archetype")

    seed_dir = best_seed_dir(model_dir(args.family, args.L, args.split, args.models_root),
                             METRIC_COL[args.family])
    days, s, a = day_labels(seed_dir, arch)
    counts = pd.crosstab(pd.Series(a, name="archetype"), pd.Series(s, name="unit"))
    counts = counts.reindex(index=range(len(arch_names)), columns=range(args.L), fill_value=0)

    # majority: archetype with most days inside the unit (used by purity);
    # matched: archetype whose days the unit captures the largest share of (used for
    # colours and column order). An unused unit (no day assigned) gets -1, sorted last.
    used = counts.values.sum(axis=0) > 0
    majority = np.where(used, counts.values.argmax(axis=0), -1)
    row_share = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    matched = np.where(used, row_share.values.argmax(axis=0), -1)
    order = sorted(range(args.L), key=lambda u: (not used[u], matched[u], -row_share.values[:, u].max()))
    counts = counts[order]
    share = row_share[order]

    score_nmi, purity = agreement(a, s)
    n_used = int(used.sum())

    label = f"{display_name(args.family)} L={args.L} ({args.split})"
    out_dir = split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis" / "archetype" / "softmax_vs_archetypes",
                            args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.family}_L{args.L}"

    # 1. contingency heatmap
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{v:.0%}\n({counts.values[i, j]})", ha="center", va="center",
                    fontsize=11, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(args.L), [f"h{u}" if used[u] else f"h{u}\n(unused)" for u in order])
    ax.set_yticks(range(len(arch_names)), arch_names)
    for tick, c in zip(ax.get_yticklabels(), ARCH_COLORS):
        tick.set_color(c)
        tick.set_fontweight("bold")
    ax.set_xlabel("Softmax unit (one state per day)")
    ax.set_ylabel("Dominant archetype")
    if show_titles():
        ax.set_title(f"{label}, best seed ({n_used} of {args.L} units used) vs archetypes (k = 5)\n"
                     f"share of each archetype's days per unit · NMI = {score_nmi:.2f} · "
                     f"purity = {purity:.0%} · {len(days)} days", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.85, label="share of archetype days")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_contingency.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2. timeline strips; a unit takes the colour of its matched archetype
    full = pd.date_range(days.min(), days.max(), freq="D")
    a_full = pd.Series(a, index=days).reindex(full)
    s_full = pd.Series(s, index=days).reindex(full)
    fig, axes = plt.subplots(2, 1, figsize=(18, 2.4), sharex=True)
    cmap = ListedColormap(ARCH_COLORS)
    cmap.set_bad("white")
    a_img = np.ma.masked_invalid(a_full.values.astype(float))[None, :]
    s_img = np.ma.masked_invalid(s_full.map(lambda u: matched[int(u)] if pd.notna(u) else np.nan)
                                 .values.astype(float))[None, :]
    extent = [mdates.date2num(full[0]), mdates.date2num(full[-1]), 0, 1]
    for ax, img, name in ((axes[0], a_img, "archetype"), (axes[1], s_img, "Softmax")):
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(ARCH_COLORS) - 0.5,
                  extent=extent, interpolation="nearest")
        ax.set_yticks([0.5], [name])
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    units_of = {k: [f"h{u}" for u in order if matched[u] == k] for k in range(len(arch_names))}
    handles = [Patch(color=c, label=f"{n}  ·  {', '.join(units_of[k]) or '-'}")
               for k, (c, n) in enumerate(zip(ARCH_COLORS, arch_names))]
    axes[0].legend(handles=handles, ncol=1, loc="upper left", bbox_to_anchor=(1.005, 1.35),
                   frameon=False, fontsize=11, handlelength=1.0,
                   title="archetype · Softmax unit", title_fontsize=11)
    axes[0].set_yticks([0.5], ["dominant\narchetype"])
    axes[1].set_yticks([0.5], ["Softmax state\n(colour of its\nmatched archetype)"])
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_timeline.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    counts.index = arch_names
    counts.columns = [f"h{u}" for u in order]
    counts.to_csv(out_dir / f"{stem}_contingency.csv")
    print(f"{label}: seed {seed_dir.name}, {len(days)} days, NMI={score_nmi:.3f}, purity={purity:.3f}")
    print(f"unit -> matched archetype: { {f'h{u}': arch_names[matched[u]] if used[u] else 'unused' for u in order} }")
    if args.sweep_L:
        sweep = seed_sweep(args.family, args.sweep_L, args.split, args.models_root, arch)
        sweep.to_csv(out_dir / f"{args.family}_seed_sweep.csv", index=False)
        print(sweep.groupby("L")[["units_with_days", "nmi", "purity"]].agg(["mean", "std"]).round(3))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
