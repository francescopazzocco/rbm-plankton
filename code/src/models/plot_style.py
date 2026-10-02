"""plot_style.py - Shared resolution and font sizes for every saved figure.

Figures end up on 1920x1080 slides, often shrunk to half the slide width or
less. What decides legibility there is the text height on the slide, which is
fontsize * (displayed width / figure width in inches), independent of dpi; dpi
only decides sharpness. Hence two separate knobs:

  FIG_DPI  - raster resolution of every PNG. 300 keeps a figure sharp when it
             is shown at up to ~2x its nominal size (HiDPI screens, 4K
             projectors) instead of being upscaled and blurred.
  FONT_RC  - default font sizes. Matplotlib's 10 pt default shrinks below
             legibility on a slide; these defaults apply wherever a call does
             not set its own fontsize.

No third-party deps beyond matplotlib, so lightweight standalone scripts can
import it without pulling in models.visualization.
"""

from __future__ import annotations

import os

import matplotlib as mpl

FIG_DPI = 300

FONT_RC = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "figure.titlesize": 15,
}


def show_titles() -> bool:
    """False when RBM_NO_TITLES=1: figure titles are dropped for the paper, where
    the caption carries the same information. Panel labels that tell panels
    apart are kept either way."""
    return os.environ.get("RBM_NO_TITLES", "") != "1"


def apply_style() -> None:
    """Set the shared font sizes and savefig dpi on matplotlib's rcParams."""
    mpl.rcParams.update(FONT_RC)
    mpl.rcParams["savefig.dpi"] = FIG_DPI
