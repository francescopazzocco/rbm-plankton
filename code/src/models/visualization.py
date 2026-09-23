"""
visualization.py - All plotting functions for the RBM plankton project.

Sections (in order):
  main_multiseed        export_results_csv, plot_training_curves,
                        plot_weight_heatmap, plot_hidden_activations
  sweep_analysis        plot_final_metric, plot_sweep_curves, plot_nb_diagnostics
  hidden_dominant_state plot_weight_profiles, plot_state_timeline
  hidden_mean_activation  plot_family
  hidden_cross_model    plot_correlation, plot_pattern_frequency,
                        plot_seasonal_profiles
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import leaves_list, linkage

from .palette import OKABE_ITO, PALETTE, get_markers, get_palette

# =============================================================================
# main_multiseed
# =============================================================================

def export_results_csv(history, W, taxa_cols, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {"epoch": history["epoch"], "train_mse": history["train_mse"]}
    if history.get("val_mse") and history["val_mse"][0] is not None:
        cols["val_mse"] = history["val_mse"]
    if history.get("train_pll"):
        cols["train_pll"] = history["train_pll"]
    if history.get("val_pll") and history["val_pll"][0] is not None:
        cols["val_pll"] = history["val_pll"]
    if history.get("train_nll"):
        cols["train_nll"] = history["train_nll"]
    if history.get("val_nll") and history["val_nll"][0] is not None:
        cols["val_nll"] = history["val_nll"]
    if history.get("theta_mean"):
        cols["theta_mean"] = history["theta_mean"]
    if history.get("pi_mean"):
        cols["pi_mean"] = history["pi_mean"]
    if history.get("sat_mid"):
        cols["sat_lo"]  = history["sat_lo"]
        cols["sat_hi"]  = history["sat_hi"]
        cols["sat_mid"] = history["sat_mid"]
    pd.DataFrame(cols).to_csv(out_dir / "rbm_training_curves.csv", index=False)
    pd.DataFrame(W, columns=[f"h{j}" for j in range(W.shape[1])],
                 index=taxa_cols).to_csv(out_dir / "rbm_weights.csv")
    print(f"[CSV]  saved training curves and weights -> {out_dir}/")


def plot_training_curves(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"],
            color="steelblue", lw=1.5, label="train MSE")
    if history.get("val_mse") and history["val_mse"][0] is not None:
        ax.plot(history["epoch"], history["val_mse"],
                color="firebrick", lw=1.5, ls="--", label="val MSE")
    if history.get("train_pll"):
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["train_pll"],
                 color="steelblue", lw=1.0, ls=":", label="train PLL", alpha=0.7)
        if history.get("val_pll") and history["val_pll"][0] is not None:
            ax2.plot(history["epoch"], history["val_pll"],
                     color="firebrick", lw=1.0, ls="-.", label="val PLL", alpha=0.7)
        ax2.set_ylabel("PLL")
        ax2.legend(loc="lower right")
    if history.get("train_nll"):
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["train_nll"],
                 color="steelblue", lw=1.0, ls=":", label="train NLL", alpha=0.7)
        if history.get("val_nll") and history["val_nll"][0] is not None:
            ax2.plot(history["epoch"], history["val_nll"],
                     color="firebrick", lw=1.0, ls="-.", label="val NLL", alpha=0.7)
        ax2.set_ylabel("NLL")
        ax2.legend(loc="lower right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Training curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = out_dir / f"training_curves{ext}"
        plt.savefig(path, dpi=150 if ext == ".png" else 300, bbox_inches="tight")
        print(f"[Plot]  saved {path}")
    plt.close()


def plot_weight_heatmap(W, taxa_cols, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    link  = linkage(W, method="ward")
    order = leaves_list(link)
    W_ord = W[order]
    vmax  = np.abs(W).max()
    fig, ax = plt.subplots(1, 1, figsize=(8, max(5, len(W_ord) * 0.15)))
    im = ax.imshow(W_ord, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"h{j}" for j in range(W.shape[1])], fontsize=9)
    ax.set_yticks(range(len(W_ord)))
    ax.set_yticklabels([taxa_cols[i] for i in order], fontsize=4)
    ax.set_title("Weight matrix W (taxa clustered)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = out_dir / f"weight_heatmap{ext}"
        plt.savefig(path, dpi=150 if ext == ".png" else 300, bbox_inches="tight")
        print(f"[Plot]  saved {path}")
    plt.close()


def plot_hidden_activations(rbm, X_train, X_val, dates_train, dates_val, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        H_all = torch.cat([rbm.hidden_probs(X_train),
                           rbm.hidden_probs(X_val)], dim=0).cpu().numpy()
    dates_all = pd.concat([dates_train, dates_val]).reset_index(drop=True)
    n_hidden = H_all.shape[1]
    colors   = get_palette(n_hidden)
    fig, axes = plt.subplots(n_hidden, 1, figsize=(13, 2.5 * n_hidden), sharex=True)
    if n_hidden == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        vals   = H_all[:, j]
        near_0 = (vals < 0.1).mean()
        near_1 = (vals > 0.9).mean()
        mid    = 1 - near_0 - near_1
        ax.plot(dates_all, vals, lw=0.5, color=colors[j % len(colors)], alpha=0.6)
        ax.plot(dates_all, pd.Series(vals).rolling(14, center=True).mean(),
                lw=1.8, color=colors[j % len(colors)])
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(h=1|v)", fontsize=8)
        ax.set_title(
            f"h{j}  |  <0.1: {near_0:.0%}   >0.9: {near_1:.0%}   "
            f"middle: {mid:.0%}  ->  {'binary' if mid < 0.15 else 'continuous'}",
            fontsize=8, loc="left"
        )
        ax.grid(True, alpha=0.25)
        for year in range(2019, 2025):
            ax.axvspan(pd.Timestamp(f"{year}-06-01"),
                       pd.Timestamp(f"{year}-09-01"),
                       alpha=0.07, color="orange")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.suptitle("Hidden unit activations h(t)  |  orange = summer", fontsize=11)
    plt.tight_layout()
    path = out_dir / "hidden_activations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")


# =============================================================================
# sweep_analysis
# =============================================================================

FAMILY_META = {
    "bernoulli_median": dict(col="val_pll", label="Val PLL", better="higher"),
    "bernoulli_zero":   dict(col="val_pll", label="Val PLL", better="higher"),
    "nb":               dict(col="val_nll", label="Val NLL", better="lower"),
    "zinb":             dict(col="val_nll", label="Val NLL", better="lower"),
    "nb_relu":          dict(col="val_nll", label="Val NLL", better="lower"),
    "zinb_relu":        dict(col="val_nll", label="Val NLL", better="lower"),
    "nb_sigmoid":       dict(col="val_nll", label="Val NLL", better="lower"),
    "nb_softmax":       dict(col="val_nll", label="Val NLL", better="lower"),
    "zinb_sigmoid":     dict(col="val_nll", label="Val NLL", better="lower"),
    "zinb_softmax":     dict(col="val_nll", label="Val NLL", better="lower"),
}

# Canonical, project-wide family order for color assignment (NOT the same as
# FAMILY_META's key order, which drives panel layout elsewhere in this file
# and is left alone). Grouping nb-profiles and zinb-profiles together here
# means the two families sharing an activation profile (e.g. nb_relu /
# zinb_relu) sit near each other in the Okabe-Ito cycle; other scripts
# (compare_model_reconstructions.py) import this list so a given family gets
# the same color in every figure across the project, not just within one.
FAMILY_ORDER = [
    "bernoulli_median", "bernoulli_zero",
    "nb", "nb_relu", "nb_sigmoid", "nb_softmax",
    "zinb", "zinb_relu", "zinb_sigmoid", "zinb_softmax",
]

# 10 families > 6 safe PALETTE colors, so colors repeat every 6th family
# (nb_sigmoid reuses bernoulli_median's color, etc). Deliberately cycling the
# 6-color PALETTE rather than escalating to the full 8-color OKABE_ITO here:
# OKABE_ITO's yellow/reddish-purple are excluded from PALETTE precisely
# because they're poor on a white line-plot background (see palette.py), and
# a washed-out yellow line is worse than an earlier repeat. Repeats are safe
# because every caller pairs color with a second channel: plot_final_metric's
# PROFILE_MARKERS (marker shape, within one nb/zinb panel) or
# compare_model_reconstructions.py's per-family linestyle (across >6 families
# in one legend).
COLORS = {family: PALETTE[i % len(PALETTE)] for i, family in enumerate(FAMILY_ORDER)}


def load_curves(csv_path: Path, col: str) -> pd.Series | None:
    """Load a metric column from a training curves CSV; return None if all NaN."""
    df = pd.read_csv(csv_path)
    if col not in df.columns or df[col].isna().all():
        return None
    return df.set_index("epoch")[col]


def _aggregate_df(csv_paths: list[Path], col: str) -> pd.DataFrame | None:
    curves = [load_curves(p, col) for p in csv_paths]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    return pd.concat(curves, axis=1)


def aggregate_curves(csv_paths: list[Path], col: str) -> tuple[pd.Series, pd.Series] | None:
    """Aggregate a metric across seeds; return (mean, std) or None if all failed."""
    df = _aggregate_df(csv_paths, col)
    if df is None:
        return None
    return df.mean(axis=1), df.std(axis=1)


def aggregate_curves_extrema(
    csv_paths: list[Path], col: str
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series] | None:
    """Aggregate a metric across seeds; return (mean, std, min, max) or None if all failed."""
    df = _aggregate_df(csv_paths, col)
    if df is None:
        return None
    return df.mean(axis=1), df.std(axis=1), df.min(axis=1), df.max(axis=1)


# Panels for plot_final_metric: each becomes its own figure/file. nb and zinb
# each bundle their 4 activation profiles (plain/relu/sigmoid/softmax) into one
# plot, distinguished by marker shape so profiles stay comparable across L.
PANEL_GROUPS = [
    ("bernoulli_median", ["bernoulli_median"]),
    ("bernoulli_zero",   ["bernoulli_zero"]),
    ("nb",               ["nb", "nb_relu", "nb_sigmoid", "nb_softmax"]),
    ("zinb",             ["zinb", "zinb_relu", "zinb_sigmoid", "zinb_softmax"]),
]

PROFILE_MARKERS = {"": "o", "relu": "^", "sigmoid": "s", "softmax": "D"}

# Hidden-unit type per PROFILE_MARKERS suffix, i.e. what varies *within* a
# PANEL_GROUPS entry (nb/nb_relu/nb_sigmoid/nb_softmax all share NB visible
# units; only the hidden-unit type differs). Used for legend labels so a line
# reads "Sigmoid hidden" instead of the bare family string "nb_sigmoid" --
# see ARCHITECTURE.md's per-family tables and DECISION_LOG LOG-020/021/022.
PROFILE_LABEL = {
    "":        "Bernoulli hidden",
    "relu":    "ReLU hidden",
    "sigmoid": "Sigmoid hidden",
    "softmax": "Softmax hidden",
}

# Visible-unit distribution per PANEL_GROUPS title key, i.e. what is fixed
# *across* a panel group. Combined with PROFILE_LABEL in the panel title so
# every plot states both halves of the model name (visible x hidden) instead
# of relying on the reader already knowing the family-name convention.
PANEL_VISIBLE_LABEL = {
    "bernoulli_median": "BB visible units (binarised at per-taxon median)",
    "bernoulli_zero":   "BB visible units (binarised at zero)",
    "nb":               "NB visible units",
    "zinb":             "ZINB visible units",
}


def _profile_of(family: str, base: str) -> str:
    return "" if family == base else family[len(base) + 1:]


# Display name for every family, "<visible>-<hidden>", used in every plot
# title, legend and tick label in place of the raw directory name
# ("nb_sigmoid" -> "NB-Sigmoid"). A bare "nb"/"zinb" family has Bernoulli
# hidden units (ARCHITECTURE.md), so it is spelled out rather than left as
# "NB", which would not say which hidden-unit type it is.
FAMILY_DISPLAY_NAME = {
    "bernoulli_median": "BB-median",
    "bernoulli_zero":   "BB-zero",
    "nb":               "NB-Bernoulli",
    "nb_relu":          "NB-ReLU",
    "nb_sigmoid":       "NB-Sigmoid",
    "nb_softmax":       "NB-Softmax",
    "zinb":             "ZINB-Bernoulli",
    "zinb_relu":        "ZINB-ReLU",
    "zinb_sigmoid":     "ZINB-Sigmoid",
    "zinb_softmax":     "ZINB-Softmax",
}


def display_name(family: str) -> str:
    """Clean plot label for a family directory name; unknown names pass through."""
    return FAMILY_DISPLAY_NAME.get(family, family)


def _plot_final_metric_panel(ax, title: str, families: list[str], runs) -> set[int]:
    """Draw one panel_groups entry (one family bundle) onto an existing axis.

    Returns the set of L values plotted, so callers can set shared xticks.
    """
    all_xs: set[int] = set()
    for family in families:
        meta = FAMILY_META[family]
        col = meta["col"]
        family_runs = runs.get(family, {})
        xs, means, stds, mins, maxs = [], [], [], [], []
        for l_val in sorted(family_runs):
            agg = aggregate_curves_extrema(family_runs[l_val], col)
            if agg is None:
                continue
            mean_curve, std_curve, min_curve, max_curve = agg
            xs.append(l_val)
            means.append(mean_curve.iloc[-1])
            stds.append(std_curve.iloc[-1])
            mins.append(min_curve.iloc[-1])
            maxs.append(max_curve.iloc[-1])
        if not xs:
            continue
        all_xs.update(xs)
        color   = COLORS[family]
        profile = _profile_of(family, title)
        marker  = PROFILE_MARKERS.get(profile, "o")
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.12, zorder=0)
        ax.errorbar(xs, means, yerr=stds, fmt=f"{marker}-", color=color,
                    linewidth=2, markersize=7, capsize=4,
                    label=PROFILE_LABEL.get(profile, family) if len(families) > 1 else display_name(family))
        if len(families) == 1:
            for x, y, s in zip(xs, means, stds):
                ax.annotate(f"{y:.3f}±{s:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)

    visible_label = PANEL_VISIBLE_LABEL.get(title, title)
    if len(families) > 1:
        panel_title = f"{visible_label} — final val metric vs L, by hidden-unit type"
    else:
        hidden_label = PROFILE_LABEL[_profile_of(families[0], title)]
        panel_title = f"{visible_label}, {hidden_label} — final val metric vs L"
    ax.set_title(panel_title, fontsize=10)
    ax.set_xlabel("L (hidden units)")
    ax.set_ylabel(FAMILY_META[families[0]]["label"])
    if all_xs:
        ax.set_xticks(sorted(all_xs))
    ax.grid(True, alpha=0.3)
    if len(families) > 1:
        ax.legend(fontsize=8)
    return all_xs


def plot_final_metric(runs, figures_dir: Path):
    for title, families in PANEL_GROUPS:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        _plot_final_metric_panel(ax, title, families, runs)
        ax.set_title(ax.get_title() + "\n(last epoch, mean ± std, shaded = min/max over seeds)",
                     fontsize=10)

        fig.tight_layout()
        out = figures_dir / f"sweep_final_metric_{title}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)


def plot_final_metric_overview(runs, figures_dir: Path):
    """Side-by-side NB / ZINB hidden-unit-type comparison: each panel fixes
    the visible distribution (NB or ZINB) and overlays all four hidden-unit
    types (Bernoulli/ReLU/Sigmoid/Softmax) tested for it, so the two count
    model families -- and the hidden-unit-type choice within each -- are
    directly comparable in one figure. Bernoulli-visible families have only
    one hidden-unit type (no ReLU/Sigmoid/Softmax variant was trained for
    them), so they are left to their own single-line plot_final_metric panels.
    """
    count_groups = [(t, f) for t, f in PANEL_GROUPS if t in ("nb", "zinb")]
    fig, axes = plt.subplots(1, len(count_groups), figsize=(6 * len(count_groups), 4.5))
    for ax, (title, families) in zip(axes, count_groups):
        _plot_final_metric_panel(ax, title, families, runs)

    fig.suptitle("Hidden-unit-type comparison — final val NLL vs L "
                 "(mean ± std, shaded = min/max over seeds)",
                 fontsize=12)
    fig.tight_layout()
    out = figures_dir / "sweep_final_metric_overview.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_final_metric_individual(runs, figures_dir: Path):
    """One standalone, annotated figure per hidden-unit type within the NB and
    ZINB groups (nb, nb_relu, nb_sigmoid, nb_softmax, zinb, zinb_relu,
    zinb_sigmoid, zinb_softmax) -- the single-line view of what plot_final_metric
    bundles into one multi-line panel per visible family. Bernoulli families
    (bernoulli_median/bernoulli_zero) have only one hidden-unit type each, so
    plot_final_metric's own panel for them already is this standalone view;
    they are not duplicated here.
    """
    out_dir = figures_dir / "individual"
    out_dir.mkdir(parents=True, exist_ok=True)
    for base, families in PANEL_GROUPS:
        if len(families) == 1:
            continue
        for family in families:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            all_xs = _plot_final_metric_panel(ax, base, [family], runs)
            if not all_xs:
                plt.close(fig)
                continue
            ax.set_title(ax.get_title() + "\n(last epoch, mean ± std, shaded = min/max over seeds)",
                         fontsize=10)
            fig.tight_layout()
            out = out_dir / f"sweep_final_metric_{family}.png"
            fig.savefig(out, dpi=150)
            print(f"Saved: {out}")
            plt.close(fig)


def plot_sweep_curves(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(15, 4), sharey=False)
    fig.suptitle("Val metric training curves by L (mean ± 1σ over seeds)", fontsize=13)
    cmap = plt.colormaps["viridis"]

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        l_values = sorted(family_runs)
        n = len(l_values)
        for i, l_val in enumerate(l_values):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is None:
                ax.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i * 0.07),
                            xycoords="axes fraction", fontsize=8, color="red")
                continue
            mean_curve, std_curve = agg
            color = cmap(i / max(n - 1, 1))
            ax.plot(mean_curve.index, mean_curve.values, color=color,
                    linewidth=1.5, label=f"L={l_val}")
            ax.fill_between(mean_curve.index,
                            mean_curve.values - std_curve.values,
                            mean_curve.values + std_curve.values,
                            color=color, alpha=0.2)
        ax.set_title(display_name(family))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(meta["label"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_nb_diagnostics(runs, figures_dir: Path):
    """Two-panel figure: NB val_nll and theta_mean trajectories per L (mean ± 1σ)."""
    nb_runs = runs.get("nb", {})
    l_values = sorted(nb_runs)
    n = len(l_values)
    cmap = plt.colormaps["viridis"]

    fig, (ax_nll, ax_theta) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NB-Bernoulli: NLL and θ trajectories by L (mean ± 1σ over seeds)", fontsize=13)

    for i, l_val in enumerate(l_values):
        color = cmap(i / max(n - 1, 1))
        label = f"L={l_val}"
        nll_agg = aggregate_curves(nb_runs[l_val], "val_nll")
        if nll_agg is not None:
            nll_mean, nll_std = nll_agg
            ax_nll.plot(nll_mean.index, nll_mean.values, color=color, linewidth=1.5, label=label)
            ax_nll.fill_between(nll_mean.index,
                                nll_mean.values - nll_std.values,
                                nll_mean.values + nll_std.values,
                                color=color, alpha=0.2)
        else:
            ax_nll.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i * 0.07),
                            xycoords="axes fraction", fontsize=8, color="red")
        theta_agg = aggregate_curves(nb_runs[l_val], "theta_mean")
        if theta_agg is not None:
            theta_mean, theta_std = theta_agg
            ax_theta.plot(theta_mean.index, theta_mean.values, color=color, linewidth=1.5, label=label)
            ax_theta.fill_between(theta_mean.index,
                                  theta_mean.values - theta_std.values,
                                  theta_mean.values + theta_std.values,
                                  color=color, alpha=0.2)

    ax_nll.set_title("Val NLL (↓)")
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.legend(fontsize=8)
    ax_nll.grid(True, alpha=0.3)
    ax_theta.set_title("θ mean (dispersion)")
    ax_theta.set_xlabel("Epoch")
    ax_theta.set_ylabel("θ")
    ax_theta.legend(fontsize=8)
    ax_theta.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_nb_diagnostics.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_zinb_diagnostics(runs, figures_dir: Path):
    """Three-panel figure: ZINB val_nll, theta_mean, and pi_mean trajectories per L."""
    zinb_runs = runs.get("zinb", {})
    l_values = sorted(zinb_runs)
    n = len(l_values)
    cmap = plt.colormaps["viridis"]

    fig, (ax_nll, ax_theta, ax_pi) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("ZINB-Bernoulli: NLL, θ, and π trajectories by L (mean ± 1σ over seeds)", fontsize=13)

    for i, l_val in enumerate(l_values):
        color = cmap(i / max(n - 1, 1))
        label = f"L={l_val}"
        nll_agg = aggregate_curves(zinb_runs[l_val], "val_nll")
        if nll_agg is not None:
            nll_mean, nll_std = nll_agg
            ax_nll.plot(nll_mean.index, nll_mean.values, color=color, linewidth=1.5, label=label)
            ax_nll.fill_between(nll_mean.index,
                                nll_mean.values - nll_std.values,
                                nll_mean.values + nll_std.values,
                                color=color, alpha=0.2)
        else:
            ax_nll.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i * 0.07),
                            xycoords="axes fraction", fontsize=8, color="red")
        theta_agg = aggregate_curves(zinb_runs[l_val], "theta_mean")
        if theta_agg is not None:
            theta_mean, theta_std = theta_agg
            ax_theta.plot(theta_mean.index, theta_mean.values, color=color, linewidth=1.5, label=label)
            ax_theta.fill_between(theta_mean.index,
                                  theta_mean.values - theta_std.values,
                                  theta_mean.values + theta_std.values,
                                  color=color, alpha=0.2)
        pi_agg = aggregate_curves(zinb_runs[l_val], "pi_mean")
        if pi_agg is not None:
            pi_mean, pi_std = pi_agg
            ax_pi.plot(pi_mean.index, pi_mean.values, color=color, linewidth=1.5, label=label)
            ax_pi.fill_between(pi_mean.index,
                               pi_mean.values - pi_std.values,
                               pi_mean.values + pi_std.values,
                               color=color, alpha=0.2)

    ax_nll.set_title("Val NLL (↓)")
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.legend(fontsize=8)
    ax_nll.grid(True, alpha=0.3)
    ax_theta.set_title("θ mean (dispersion)")
    ax_theta.set_xlabel("Epoch")
    ax_theta.set_ylabel("θ")
    ax_theta.legend(fontsize=8)
    ax_theta.grid(True, alpha=0.3)
    ax_pi.set_title("π mean (inflation)")
    ax_pi.set_xlabel("Epoch")
    ax_pi.set_ylabel("π")
    ax_pi.legend(fontsize=8)
    ax_pi.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_zinb_diagnostics.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# hidden_dominant_state
# =============================================================================

TOP_SPECIES_PER_UNIT = 8


def load_weight_matrix(csv: Path) -> pd.DataFrame:
    """Return weight DataFrame (species x hidden units) from rbm_weights.csv."""
    return pd.read_csv(csv, index_col=0)


def select_top_species(W: pd.DataFrame) -> pd.DataFrame:
    """Keep union of top-N species per hidden unit, sorted by dominant unit."""
    top_idx = set()
    for col in W.columns:
        top_idx.update(W[col].abs().nlargest(TOP_SPECIES_PER_UNIT).index)
    W_filtered = W.loc[sorted(top_idx)]
    dominant = W_filtered.abs().values.argmax(axis=1)
    order = np.argsort(dominant, kind="stable")
    return W_filtered.iloc[order]


def load_activations(csv: Path) -> pd.DataFrame:
    """Load hidden activations CSV indexed by date (io.load_hidden_activations)."""
    from .io import load_hidden_activations
    return load_hidden_activations(csv)


def dominant_state(activations: pd.DataFrame) -> pd.Series:
    """Assign each sample to the hidden unit with highest activation."""
    return pd.Series(
        activations.values.argmax(axis=1),
        index=activations.index,
        name="dominant"
    )


PATTERN_THRESHOLD = 0.5


def hidden_binary(activations: pd.DataFrame, mode: str = "threshold",
                  threshold: float = PATTERN_THRESHOLD) -> pd.DataFrame:
    """Discretise hidden activations into a binary pattern, one bit per unit.

    mode="threshold"  unit is on when its activation >= threshold.  Intended for
                      Bernoulli and sigmoid hidden units.
    mode="winner"     one-hot on the argmax unit.  Intended for softmax units,
                      whose activations sum to 1 so that a fixed cut is
                      meaningless (at L>=10 every unit sits below 0.5).

    Known weakness of mode="threshold": a unit whose activation never crosses
    the cut contributes a constant bit, so the resulting pattern set can be an
    artefact of the threshold rather than a property of the model.  Tracked as
    item 17 in .claude/REORG_AND_VALIDATION.md (Phase B).
    """
    values = activations.to_numpy(dtype=float)

    if mode == "threshold":
        binary = (values >= threshold).astype(np.int8)
    elif mode == "winner":
        binary = np.zeros_like(values, dtype=np.int8)
        binary[np.arange(len(values)), values.argmax(axis=1)] = 1
    else:
        raise ValueError(f"Unknown mode={mode!r}; expected 'threshold' or 'winner'.")

    return pd.DataFrame(binary, columns=activations.columns, index=activations.index)


def pattern_labels(binary: pd.DataFrame) -> pd.Series:
    """Binary-string label per row, e.g. '010101'.

    Leftmost digit is the first hidden unit (h0), rightmost the last.
    """
    return binary.astype(int).astype(str).agg("".join, axis=1).rename("pattern")


def pattern_frequency(binary: pd.DataFrame) -> pd.DataFrame:
    """Count how often each binary pattern occurs, most frequent first."""
    labels = pattern_labels(binary)
    counts = labels.value_counts()
    df = pd.DataFrame({
        "pattern":  counts.index,
        "n_days":   counts.values,
        "fraction": (counts.values / len(labels)).round(4),
    })
    df["n_units_on"] = df["pattern"].apply(lambda p: p.count("1"))
    return df.reset_index(drop=True)


def plot_weight_profiles(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)
    fig, axes = plt.subplots(1, n_l, figsize=(4 * n_l, 7), sharey=False)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(
        f"{display_name(family)} - weight profiles (top-{TOP_SPECIES_PER_UNIT} species per unit)",
        fontsize=11
    )
    for ax, l_val in zip(axes, l_values):
        W = load_weight_matrix(family_runs[l_val]["weights"])
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
    out = out_dir / f"{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_state_timeline(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)
    fig, axes = plt.subplots(n_l, 1, figsize=(14, 2.2 * n_l), sharex=True)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(f"{display_name(family)} - dominant hidden state over time", fontsize=11)
    max_l  = max(l_values)
    unit_colors = get_palette(max_l)

    for ax, l_val in zip(axes, l_values):
        act   = load_activations(family_runs[l_val]["activations"])
        state = dominant_state(act)
        dates = state.index
        colors = [unit_colors[s] for s in state.values]
        ax.scatter(dates, np.zeros(len(dates)), c=colors,
                   marker="|", s=200, linewidths=2)
        ax.set_yticks([])
        ax.set_ylabel(f"L={l_val}", rotation=0, labelpad=30, fontsize=9, va="center")
        ax.set_xlim(dates.min(), dates.max())
        handles = [
            plt.Line2D([0], [0], marker="|", color="w",
                       markerfacecolor=unit_colors[i],
                       markeredgecolor=unit_colors[i],
                       markersize=10, label=f"h{i}")
            for i in range(l_val)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=l_val, framealpha=0.7)

    axes[-1].set_xlabel("date", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# hidden_mean_activation
# =============================================================================

ABSORBER_HI = 0.90
ABSORBER_LO = 0.10

# Semantic 3-way color: always-on / always-off / active. Okabe-Ito vermillion
# and blue (not tab10 red/blue) for consistency with the rest of the project;
# gray is colorblind-neutral by construction.
_COLOR_ALWAYS_ON  = OKABE_ITO[6]  # vermillion
_COLOR_ALWAYS_OFF = "#7f7f7f"     # neutral gray
_COLOR_ACTIVE     = OKABE_ITO[5]  # blue


def mean_activations(csv: Path) -> pd.Series:
    """Mean activation per hidden unit over all samples."""
    return load_activations(csv).mean()


def plot_family(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n = len(l_values)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{display_name(family)} - mean hidden activation per unit", fontsize=12)

    for ax, l_val in zip(axes, l_values):
        means = mean_activations(family_runs[l_val])
        units = np.arange(len(means))
        bar_colors = [
            _COLOR_ALWAYS_ON if v >= ABSORBER_HI else
            _COLOR_ALWAYS_OFF if v <= ABSORBER_LO else
            _COLOR_ACTIVE
            for v in means.values
        ]
        ax.bar(units, means.values, color=bar_colors)
        ax.axhline(ABSORBER_HI, color=_COLOR_ALWAYS_ON, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(ABSORBER_LO, color=_COLOR_ALWAYS_OFF, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"L={l_val}", fontsize=10)
        ax.set_xlabel("hidden unit")
        ax.set_xticks(units)
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("mean p(h=1)")
        for i, v in enumerate(means.values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    legend = [
        mpatches.Patch(color=_COLOR_ALWAYS_ON, label=f"always-on  (>{ABSORBER_HI})"),
        mpatches.Patch(color=_COLOR_ALWAYS_OFF, label=f"always-off (<{ABSORBER_LO})"),
        mpatches.Patch(color=_COLOR_ACTIVE, label="active"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out = out_dir / f"{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# hidden_cross_model
# =============================================================================

def plot_correlation(corr: pd.DataFrame, out_dir: Path, target_l: int = 6, family: str = "nb"):
    label  = display_name(family)
    suffix = "_" + p if (p := _profile_of(family, "nb")) else ""
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = corr.abs().values.max()
    im = ax.imshow(corr.values.astype(float), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([f"BB {c}" for c in corr.columns], fontsize=9)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels([f"{label} {r}" for r in corr.index], fontsize=9)
    ax.set_xlabel("bernoulli_median units", fontsize=10)
    ax.set_ylabel(f"{label} units", fontsize=10)
    ax.set_title(f"{label} vs BB-median activation correlation  (L={target_l})", fontsize=11)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = float(corr.iloc[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    stem = f"{suffix[1:]}_L{target_l}" if suffix else f"L{target_l}"
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_pattern_frequency(freq: pd.DataFrame, out_dir: Path, target_l: int = 6,
                           family: str = "nb",
                           coverage_thresholds: tuple[float, ...] = (0.8, 0.9)):
    """Pareto-style plot: bars are per-pattern frequency (ranked, all patterns
    shown), overlaid with a cumulative-coverage line marking how many distinct
    patterns are needed to account for X% of observed days -- analogous to a
    PCA scree plot's cumulative-explained-variance cutoff.
    """
    label  = display_name(family)
    suffix = "_" + p if (p := _profile_of(family, "nb")) else ""
    freq = freq.sort_values("fraction", ascending=False).reset_index(drop=True)
    cumulative = freq["fraction"].cumsum().clip(upper=1.0)
    n = len(freq)

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.32), 5))
    # n_units_on in [0, target_l] for an L-bit pattern -- target_l+1 distinct
    # values. get_palette(target_l+1) escalates from the 6-color safe subset
    # to the full 8-color Okabe-Ito set once target_l+1 > 6, so L<=7 gets a
    # color per category with no collisions. A hand-cycled 6-color-only list
    # (this module's earlier approach) collides once target_l+1 > 6 -- e.g.
    # at L=7, "6 of 7 units on" would silently wrap onto the same black used
    # for "0 units on", reading as a meaningful color when it's an accident.
    n_units_colors = get_palette(target_l + 1)
    ax1.bar(range(n), freq["fraction"],
            color=[n_units_colors[int(u)] for u in freq["n_units_on"]])
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(freq["pattern"], fontsize=7, rotation=90,
                        fontfamily="monospace")
    ax1.set_ylabel("fraction of days (this pattern)")
    ax1.set_xlabel(f"binary pattern (h0...h{target_l - 1}, 1=ON), ranked by frequency")
    ax1.set_title(f"{label} L={target_l} - activation-pattern coverage",
                  fontweight="bold", pad=22)
    ax1.text(0.5, 1.015,
             rf"unit on iff $p(h_j=1 \mid v) \geq {PATTERN_THRESHOLD}$, "
             f"{n} distinct patterns of {2 ** target_l} possible",
             transform=ax1.transAxes, ha="center", va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(range(n), cumulative, color=OKABE_ITO[0], marker="o", ms=3, lw=1.5,
             label="cumulative coverage")
    ax2.set_ylabel("cumulative fraction of days")
    ax2.set_ylim(0, 1.05)

    cutoff_color = OKABE_ITO[6]  # vermillion
    for i, thr in enumerate(coverage_thresholds):
        k = int((cumulative >= thr).idxmax()) + 1
        ax2.axhline(thr, color=cutoff_color, ls="--", lw=0.8, alpha=0.5)
        ax2.axvline(k - 1, color=cutoff_color, ls="--", lw=0.8, alpha=0.5)
        ax2.annotate(f"{k} patterns -> {thr:.0%}",
                     xy=(k - 1, thr), xytext=(k - 1 + n * 0.02, thr - 0.06 - 0.08 * i),
                     fontsize=8, color=cutoff_color)

    fig.tight_layout()
    stem = f"{suffix[1:]}_L{target_l}" if suffix else f"L{target_l}"
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_seasonal_profiles(nb_prof: pd.DataFrame, bb_prof: pd.DataFrame,
                           out_dir: Path, target_l: int = 6, family: str = "nb"):
    label  = display_name(family)
    suffix = "_" + p if (p := _profile_of(family, "nb")) else ""
    n_units = nb_prof.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    months = range(1, 13)
    unit_colors  = get_palette(n_units)
    unit_markers = get_markers(n_units)

    for ax, prof, title in zip(axes,
                                [nb_prof, bb_prof],
                                [f"{label}-RBM  (L={target_l})",
                                 f"BB-median  (L={target_l})"]):
        for j, col in enumerate(prof.columns):
            y = prof.loc[months, col]
            # faint dashed connector + solid marker-only scatter (same style as
            # fig3's per-year seasonal-shape panel), so series read by marker
            # shape rather than relying on color alone.
            ax.plot(months, y, linestyle="--", lw=1.0, color=unit_colors[j], alpha=0.3)
            ax.plot(months, y, marker=unit_markers[j], linestyle="none",
                    color=unit_colors[j], markersize=7, alpha=0.95, label=col)
        ax.set_ylabel("mean P(h=1|v)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=7, ncol=n_units)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xticks(months)
    axes[-1].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                              fontsize=8)
    axes[-1].set_xlabel("month")
    fig.suptitle(f"Seasonal activation profiles - {label} vs BB-median", fontsize=11)
    fig.tight_layout()
    stem = f"{suffix[1:]}_L{target_l}" if suffix else f"L{target_l}"
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
