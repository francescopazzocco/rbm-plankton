"""
visualization.py - All plotting functions for the RBM plankton project.

Sections (in order):
  main_multiseed        export_results_csv, plot_training_curves,
                        plot_weight_heatmap, plot_hidden_activations
  sweep_analysis        plot_final_metric, plot_sweep_curves, plot_nb_diagnostics
  hidden_coactivation   plot_weight_profiles, plot_state_timeline
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
    colors   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
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
    "bernoulli_median": dict(col="val_pll", label="PLL (↑)", better="higher"),
    "bernoulli_zero":   dict(col="val_pll", label="PLL (↑)", better="higher"),
    "nb":               dict(col="val_nll", label="NLL (↓)", better="lower"),
}

COLORS = {
    "bernoulli_median": "#1f77b4",
    "bernoulli_zero":   "#ff7f0e",
    "nb":               "#2ca02c",
}


def load_curves(csv_path: Path, col: str) -> pd.Series | None:
    """Load a metric column from a training curves CSV; return None if all NaN."""
    df = pd.read_csv(csv_path)
    if col not in df.columns or df[col].isna().all():
        return None
    return df.set_index("epoch")[col]


def aggregate_curves(csv_paths: list[Path], col: str) -> tuple[pd.Series, pd.Series] | None:
    """Aggregate a metric across seeds; return (mean, std) or None if all failed."""
    curves = [load_curves(p, col) for p in csv_paths]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    df = pd.concat(curves, axis=1)
    return df.mean(axis=1), df.std(axis=1)


def plot_final_metric(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(14, 4), sharey=False)
    fig.suptitle("Final val metric vs L (last epoch, mean ± std over seeds)", fontsize=13)

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        xs, means, stds = [], [], []
        for l_val in sorted(family_runs):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is None:
                continue
            mean_curve, std_curve = agg
            xs.append(l_val)
            means.append(mean_curve.iloc[-1])
            stds.append(std_curve.iloc[-1])
        color = COLORS[family]
        ax.errorbar(xs, means, yerr=stds, fmt="o-", color=color, linewidth=2,
                    markersize=7, capsize=4, label=family)
        for x, y, s in zip(xs, means, stds):
            ax.annotate(f"{y:.3f}±{s:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)
        ax.set_title(family)
        ax.set_xlabel("L (hidden units)")
        ax.set_ylabel(meta["label"])
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_final_metric.png"
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
        ax.set_title(family)
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
    fig.suptitle("NB-RBM: NLL and θ trajectories by L (mean ± 1σ over seeds)", fontsize=13)

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


# =============================================================================
# hidden_coactivation
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
    """Load hidden activations CSV indexed by date."""
    return pd.read_csv(csv, index_col="date", parse_dates=True)


def dominant_state(activations: pd.DataFrame) -> pd.Series:
    """Assign each sample to the hidden unit with highest activation."""
    return pd.Series(
        activations.values.argmax(axis=1),
        index=activations.index,
        name="dominant"
    )


def plot_weight_profiles(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)
    fig, axes = plt.subplots(1, n_l, figsize=(4 * n_l, 7), sharey=False)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(
        f"{family} - weight profiles (top-{TOP_SPECIES_PER_UNIT} species per unit)",
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
    out = out_dir / f"weight_profiles_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_state_timeline(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n_l = len(l_values)
    fig, axes = plt.subplots(n_l, 1, figsize=(14, 2.2 * n_l), sharex=True)
    if n_l == 1:
        axes = [axes]
    fig.suptitle(f"{family} - dominant hidden state over time", fontsize=11)
    max_l = max(l_values)
    cmap  = plt.colormaps["tab10"]

    for ax, l_val in zip(axes, l_values):
        act   = load_activations(family_runs[l_val]["activations"])
        state = dominant_state(act)
        dates = state.index
        colors = [cmap(s / max_l) for s in state.values]
        ax.scatter(dates, np.zeros(len(dates)), c=colors,
                   marker="|", s=200, linewidths=2)
        ax.set_yticks([])
        ax.set_ylabel(f"L={l_val}", rotation=0, labelpad=30, fontsize=9, va="center")
        ax.set_xlim(dates.min(), dates.max())
        handles = [
            plt.Line2D([0], [0], marker="|", color="w",
                       markerfacecolor=cmap(i / max_l),
                       markeredgecolor=cmap(i / max_l),
                       markersize=10, label=f"h{i}")
            for i in range(l_val)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  ncol=l_val, framealpha=0.7)

    axes[-1].set_xlabel("date", fontsize=9)
    fig.tight_layout()
    out = out_dir / f"state_timeline_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# hidden_mean_activation
# =============================================================================

ABSORBER_HI = 0.90
ABSORBER_LO = 0.10


def mean_activations(csv: Path) -> pd.Series:
    """Mean activation per hidden unit over all samples."""
    return pd.read_csv(csv, index_col="date").mean()


def plot_family(family: str, family_runs: dict, out_dir: Path):
    l_values = sorted(family_runs)
    n = len(l_values)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{family} - mean hidden activation per unit", fontsize=12)

    for ax, l_val in zip(axes, l_values):
        means = mean_activations(family_runs[l_val])
        units = np.arange(len(means))
        bar_colors = [
            "#d62728" if v >= ABSORBER_HI else
            "#7f7f7f" if v <= ABSORBER_LO else
            "#1f77b4"
            for v in means.values
        ]
        ax.bar(units, means.values, color=bar_colors)
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

    legend = [
        mpatches.Patch(color="#d62728", label=f"always-on  (>{ABSORBER_HI})"),
        mpatches.Patch(color="#7f7f7f", label=f"always-off (<{ABSORBER_LO})"),
        mpatches.Patch(color="#1f77b4", label="active"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out = out_dir / f"mean_activation_{family}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# hidden_cross_model
# =============================================================================

def plot_correlation(corr: pd.DataFrame, out_dir: Path, target_l: int = 6):
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = corr.abs().values.max()
    im = ax.imshow(corr.values.astype(float), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([f"BB {c}" for c in corr.columns], fontsize=9)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels([f"NB {r}" for r in corr.index], fontsize=9)
    ax.set_xlabel("bernoulli_median units", fontsize=10)
    ax.set_ylabel("NB units", fontsize=10)
    ax.set_title(f"NB vs BB-median activation correlation  (L={target_l})", fontsize=11)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = float(corr.iloc[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    out = out_dir / "cross_model_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_pattern_frequency(freq: pd.DataFrame, out_dir: Path, target_l: int = 6):
    top = freq.head(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(top)), top["fraction"],
                  color=plt.cm.tab10(top["n_units_on"] / 6))
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["pattern"], fontsize=8, rotation=45, ha="right",
                       fontfamily="monospace")
    ax.set_ylabel("fraction of days")
    ax.set_title(f"NB L={target_l} - most common 6-bit activation patterns "
                 f"(threshold=0.5, top 15 of {len(freq)} distinct)")
    ax.set_xlabel("binary pattern  (h0...h5, 1=ON)")
    for bar, row in zip(bars, top.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{row.n_days}d", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = out_dir / "nb_pattern_frequency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_seasonal_profiles(nb_prof: pd.DataFrame, bb_prof: pd.DataFrame,
                           out_dir: Path, target_l: int = 6):
    n_units = nb_prof.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    months = range(1, 13)
    cmap = plt.cm.tab10

    for ax, prof, title in zip(axes,
                                [nb_prof, bb_prof],
                                [f"NB-RBM  (L={target_l})",
                                 f"BB-median  (L={target_l})"]):
        for j, col in enumerate(prof.columns):
            ax.plot(months, prof.loc[months, col],
                    marker="o", ms=4, lw=1.5,
                    color=cmap(j / n_units), label=col)
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
    fig.suptitle("Seasonal activation profiles - NB vs BB-median", fontsize=11)
    fig.tight_layout()
    out = out_dir / "seasonal_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
