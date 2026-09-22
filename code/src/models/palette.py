"""palette.py - Shared colorblind-safe categorical palette (Okabe & Ito, 1996).

Deuteranopia/protanopia/tritanopia-safe. Use this for a small, FIXED set of
named categories (years, model families, activation profiles, hidden-unit
index, split names). Never use it for a continuous/sequential/diverging
quantity (weight magnitude, epoch index, a swept L value) -- a perceptually
uniform colormap (viridis, RdBu_r) is still the right tool there and is out
of scope for this module.

Reference: https://jfly.uni-koeln.de/color/

No third-party deps (no torch, no scipy) so this can be imported by
lightweight standalone scripts (e.g. code/scripts/train/dataset_analysis.py)
without pulling in the rest of models.visualization.
"""

from __future__ import annotations

# Full 8-color Okabe-Ito set, canonical order.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Preferred 6-color subset for <=6 categories: skips yellow (poor contrast on
# white) and reddish-purple (can read close to blue for some viewers).
PALETTE = [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[6], OKABE_ITO[5]]

# Marker cycle, deliberately longer (10) than the color cycles (6 or 8) so
# that once colors start repeating (>6 or >8 categories) the paired marker
# has *not* just repeated too -- color and marker collide only every
# lcm(len(colors), len(MARKERS)) categories instead of every len(colors).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8"]


def get_palette(n: int) -> list[str]:
    """n colorblind-safe hex colors, cycling if n exceeds the base set.

    n<=6 draws from PALETTE (the 6-color subset); n<=8 pulls in the full
    OKABE_ITO set; beyond 8, colors repeat -- pair with get_markers (or an
    explicit legend) since color alone is not a reliable encoding past 8
    categories for a colorblind viewer.
    """
    base = PALETTE if n <= len(PALETTE) else OKABE_ITO
    return [base[i % len(base)] for i in range(n)]


def get_markers(n: int) -> list[str]:
    """n marker shapes, cycling through MARKERS."""
    return [MARKERS[i % len(MARKERS)] for i in range(n)]


def style_for(label, order: list) -> tuple[str, str]:
    """(color, marker) for `label`, keyed by its fixed position in `order`.

    Pass the SAME `order` list (the full, fixed vocabulary of possible
    labels, e.g. every model family this project trains) from every call
    site that plots a subset of it, so a given label -- "nb_relu", "2023",
    "shuffled" -- gets the identical (color, marker) in every figure it
    appears in, regardless of which other labels are plotted alongside it
    in a particular figure.
    """
    idx = order.index(label)
    n = len(order)
    return get_palette(n)[idx], get_markers(n)[idx]


def style_map(order: list) -> dict:
    """{label: (color, marker)} for every label in `order`, in one call."""
    colors = get_palette(len(order))
    markers = get_markers(len(order))
    return {label: (colors[i], markers[i]) for i, label in enumerate(order)}
