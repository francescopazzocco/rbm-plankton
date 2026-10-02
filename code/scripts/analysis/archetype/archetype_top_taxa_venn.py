"""archetype_top_taxa_venn.py - Five-set Venn diagram of the top taxa of each archetype.

Each archetype (row of prof/archetypes_k5_profiles.csv) contributes the set of
its --top largest-weight taxa. The sets are drawn as the symmetric five-ellipse
Venn diagram, and every taxon is written inside the region that matches its
membership pattern (e.g. a taxon in the top set of A3 and A5 only is written
where the A3 and A5 ellipses, and no other, overlap).

Labels are placed at the point of each region farthest from its border, found
on a raster of the diagram, so they stay inside even the thin regions.

Usage:
    python code/scripts/analysis/archetype/archetype_top_taxa_venn.py [--top 5]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.ndimage import distance_transform_edt

from models.palette import ARCHETYPE_COLORS
from models.paths import DIAGNOSTIC_ROOT
from models.paths import PROJECT_ROOT as ROOT
from models.plot_style import FIG_DPI, apply_style

apply_style()

# Symmetric five-ellipse Venn layout: (centre x, centre y, width, height, angle deg).
# Every one of the 31 membership patterns has a non-empty region.
VENN5 = [
    (0.428, 0.449, 0.87, 0.50, 155.0),
    (0.469, 0.543, 0.87, 0.50, 82.0),
    (0.558, 0.523, 0.87, 0.50, 10.0),
    (0.578, 0.432, 0.87, 0.50, 118.0),
    (0.489, 0.383, 0.87, 0.50, 46.0),
]
# Where each set name goes, just outside its ellipse.
NAME_POS = [(-0.04, 0.80), (0.50, 1.07), (1.07, 0.80), (0.95, -0.07), (0.08, -0.07)]
# Community names used on the slides (archetype_rbm_comparison.md); a set not
# listed here is named after its largest-weight taxon.
COMMUNITY = {"A1": "Dinobryon", "A2": "Aulacoseira", "A3": "Cryptophyte",
             "A4": "Green algae + cyano", "A5": "Centric diatom"}

RASTER = 800


def inside(x: np.ndarray, y: np.ndarray, cx, cy, w, h, angle) -> np.ndarray:
    """Boolean mask of the points (x, y) inside one ellipse."""
    t = np.deg2rad(angle)
    dx, dy = x - cx, y - cy
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / (w / 2)) ** 2 + (v / (h / 2)) ** 2 <= 1.0


def region_anchor(pattern: tuple[bool, ...], masks: list[np.ndarray],
                  grid: np.ndarray) -> tuple[float, float]:
    """Point of the region with this membership pattern farthest from its border."""
    region = np.ones_like(masks[0])
    for m, member in zip(masks, pattern):
        region &= m if member else ~m
    dist = distance_transform_edt(region)
    iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
    return grid[ix], grid[iy]


def pretty(taxon: str) -> str:
    short = {"cyanobacteria_colonial_probably": "cyano colonial",
             "chlorophyte_colonial_dividing": "chloro. colonial"}
    return short.get(taxon, taxon.replace("_", " "))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archetypes", type=Path, default=ROOT / "prof" / "archetypes_k5_profiles.csv")
    parser.add_argument("--top", type=int, default=5, help="Taxa per archetype (largest weights)")
    parser.add_argument("--out", type=Path,
                        default=DIAGNOSTIC_ROOT / "02_model_analysis" / "archetype" / "top_taxa_venn")
    args = parser.parse_args()

    profiles = pd.read_csv(args.archetypes, index_col=0)
    names = list(profiles.index)
    if len(names) != len(VENN5):
        raise ValueError(f"layout is for {len(VENN5)} sets, got {len(names)} archetypes")
    top = {a: list(profiles.loc[a].sort_values(ascending=False).head(args.top).index) for a in names}
    lead = {a: COMMUNITY.get(a, pretty(top[a][0]).capitalize()) for a in names}

    taxa = sorted({t for ts in top.values() for t in ts})
    membership: dict[tuple[bool, ...], list[str]] = {}
    for t in taxa:
        membership.setdefault(tuple(t in top[a] for a in names), []).append(t)

    grid = np.linspace(0, 1, RASTER)
    gx, gy = np.meshgrid(grid, grid)
    masks = [inside(gx, gy, *e) for e in VENN5]

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ARCHETYPE_COLORS
    for (cx, cy, w, h, ang), c, name, pos in zip(VENN5, colors, names, NAME_POS):
        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fc=c, ec=c, alpha=0.18, lw=2))
        ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fc="none", ec=c, lw=2))
        ax.text(*pos, f"{name}\n{lead[name]}", color=c, fontsize=20, fontweight="bold",
                ha="center", va="center")

    for pattern, members in membership.items():
        x, y = region_anchor(pattern, masks, grid)
        shared = sum(pattern)
        ax.text(x, y, "\n".join(pretty(t) for t in members), ha="center", va="center",
                fontsize=15 if len(members) == 1 else 13,
                fontweight="bold" if shared > 1 else "normal")

    ax.set_xlim(-0.22, 1.24)
    ax.set_ylim(-0.13, 1.13)
    ax.set_aspect("equal")
    ax.axis("off")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"top{args.top}_venn.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    rows = [{"taxon": t, **{a: t in top[a] for a in names}, "n_archetypes": sum(t in top[a] for a in names)}
            for t in taxa]
    pd.DataFrame(rows).sort_values("n_archetypes", ascending=False).to_csv(
        args.out / f"top{args.top}_membership.csv", index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
