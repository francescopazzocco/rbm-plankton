"""
data_analysis.py — Exploratory analysis of Lake Greifen plankton time series
=============================================================================

Generates all figures for DATA_ANALYSIS.md:
  fig1_rowsum_timeseries.png  — daily row sum + 30-day rolling median
  fig2_lombscargle.png        — Lomb-Scargle periodogram on row sum
  fig3_annual_seasonal.png    — annual statistics + seasonal shape per year
  fig4_distributions.png      — marginal distributions: raw → log → z-score
  fig5_nan_structure.png      — NaN block structure over time

Usage:
  python data_analysis.py                          # uses default DATA_PATH
  python data_analysis.py path/to/data.csv         # custom path
  python data_analysis.py path/to/data.csv ./figs  # custom output dir

All figures are saved to OUTPUT_DIR (default: current directory).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import lombscargle, find_peaks
from scipy import stats

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DATA_PATH  = "data/raw/TimeSeries_countsuL_clean.csv"
OUTPUT_DIR = "."

# ε for log-transform: 10% of the global minimum non-zero value
# (open question — see DECISIONS.md; this is a reasonable starting point)
EPS_FRACTION = 0.1


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_clean(path: str):
    """
    Load CSV, drop all-zero rows and NaN rows.
    Returns clean DataFrame + list of taxa column names.
    NaN rows are returned separately for later use as test set.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa_cols = [c for c in df.columns if c != "date"]

    # Step 1 — drop all-zero rows (instrument downtime artifacts)
    nonzero_mask = df[taxa_cols].sum(axis=1) > 0
    df_nonzero   = df[nonzero_mask].copy()

    # Step 2 — separate NaN rows (ML classifier outage blocks → test set)
    nan_mask      = df_nonzero[taxa_cols].isna().any(axis=1)
    df_nan_rows   = df_nonzero[nan_mask].copy()
    df_clean      = df_nonzero[~nan_mask].copy().reset_index(drop=True)

    print(f"Raw rows          : {len(df)}")
    print(f"After drop zeros  : {len(df_nonzero)}  ({nonzero_mask.sum()} kept)")
    print(f"NaN rows (test)   : {len(df_nan_rows)}")
    print(f"Clean train rows  : {len(df_clean)}")

    return df_clean, df_nan_rows, df, taxa_cols


# ---------------------------------------------------------------------------
# Figure 1 — Row sum time series
# ---------------------------------------------------------------------------

def fig1_rowsum_timeseries(df_clean, taxa_cols, out_dir):
    rs   = df_clean.set_index("date")[taxa_cols].sum(axis=1)
    roll = rs.rolling("30D", center=True).median()
    cv   = rs.std() / rs.mean()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(rs.index, rs.values, lw=0.5, color="steelblue",
                 alpha=0.7, label="daily row sum")
    axes[0].plot(roll.index, roll.values, lw=2, color="firebrick",
                 label="30-day rolling median")
    axes[0].set_ylabel("Row sum (organisms/μL)")
    axes[0].set_title(
        f"Total daily abundance  —  CV={cv:.2f}: variability dominated by biology, not instrument drift"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(rs.index, rs.values, lw=0.5, color="steelblue", alpha=0.7)
    axes[1].semilogy(roll.index, roll.values, lw=2, color="firebrick")
    axes[1].set_ylabel("Row sum (log scale)")
    axes[1].set_title("Log scale — low-value winter structure visible")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_rowsum_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Lomb-Scargle periodogram
# ---------------------------------------------------------------------------

def fig2_lombscargle(df_clean, taxa_cols, out_dir):
    t = (df_clean["date"] - df_clean["date"].iloc[0]).dt.days.values.astype(float)
    y = df_clean[taxa_cols].sum(axis=1).values
    y = y - y.mean()

    total_days = t[-1] - t[0]
    freqs  = np.linspace(1.0 / total_days, 0.5, 5000)
    pgram  = lombscargle(t, y, 2 * np.pi * freqs, normalize=True)

    peaks, _ = find_peaks(pgram, height=0.01)
    top      = peaks[np.argsort(pgram[peaks])[::-1][:6]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    # Full spectrum
    axes[0].plot(freqs, pgram, lw=0.7, color="steelblue")
    for p in top:
        axes[0].axvline(freqs[p], color="firebrick", lw=0.9, alpha=0.7)
        axes[0].text(freqs[p], pgram[p] + 0.005, f"{1/freqs[p]:.0f}d",
                     fontsize=7, color="firebrick", ha="center")
    axes[0].set_xlabel("Frequency (cycles/day)")
    axes[0].set_ylabel("Normalised power")
    axes[0].set_title("Lomb-Scargle periodogram — full spectrum")
    axes[0].grid(True, alpha=0.3)

    # Zoom: periods > 30 days
    mask = freqs < 1 / 30
    axes[1].plot(freqs[mask], pgram[mask], lw=0.9, color="steelblue")
    for p in top:
        if freqs[p] < 1 / 30:
            axes[1].axvline(freqs[p], color="firebrick", lw=0.9, alpha=0.7)
            axes[1].text(freqs[p], pgram[p] + 0.005, f"{1/freqs[p]:.0f}d",
                         fontsize=8, color="firebrick", ha="center")
    axes[1].set_xlabel("Frequency (cycles/day)")
    axes[1].set_ylabel("Normalised power")
    axes[1].set_title(
        "Zoom: periods > 30 days  —  dominant ~365d; 1740d peak = artifact of 2022-23 extremes"
    )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_lombscargle.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Annual statistics + seasonal shape
# ---------------------------------------------------------------------------

def fig3_annual_seasonal(df_clean, taxa_cols, out_dir):
    df_clean = df_clean.copy()
    df_clean["row_sum"] = df_clean[taxa_cols].sum(axis=1)
    df_clean["year"]    = df_clean["date"].dt.year
    df_clean["month"]   = df_clean["date"].dt.month

    annual  = df_clean.groupby("year")["row_sum"].agg(["median", "mean", "std"])
    monthly = df_clean.groupby(["year", "month"])["row_sum"].median().unstack(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: annual stats
    years = annual.index
    ax = axes[0]
    ax.fill_between(years,
                    annual["mean"] - annual["std"],
                    annual["mean"] + annual["std"],
                    alpha=0.2, color="steelblue")
    ax.plot(years, annual["mean"],   "o-",  color="steelblue",
            lw=1.5, label="annual mean")
    ax.plot(years, annual["median"], "s--", color="firebrick",
            lw=1.5, label="annual median")
    ax.set_title("Annual statistics — median stable, mean inflated by 2022-23 extremes")
    ax.set_ylabel("Row sum (organisms/μL)")
    ax.set_xticks(years)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: seasonal shape per year (log scale)
    ax2    = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    for i, yr in enumerate([2019, 2020, 2021, 2022, 2023, 2024]):
        if yr in monthly.columns:
            ax2.semilogy(monthly.index, monthly[yr], "o-",
                         color=colors[i], lw=1.2, alpha=0.85, label=str(yr))
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_labels)
    ax2.set_title("Seasonal shape per year — Jan-Feb 2023 anomaly (2 orders of magnitude above peers)")
    ax2.set_ylabel("Monthly median row sum (log scale)")
    ax2.legend(title="Year")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_annual_seasonal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Marginal distributions: raw → log → z-score
# ---------------------------------------------------------------------------

def fig4_distributions(df_clean, taxa_cols, out_dir, eps_fraction=EPS_FRACTION):
    # ε = fraction of global minimum non-zero value
    min_nonzero = df_clean[taxa_cols].replace(0, np.nan).min().min()
    eps = min_nonzero * eps_fraction
    print(f"  log-transform ε = {eps:.2e}  (min nonzero = {min_nonzero:.2e})")

    selected = [
        ("aulacoseira", "dominant  | ~3% zeros"),
        ("cryptophyte", "common    | ~0% zeros"),
        ("rotifer",     "intermed. | ~0% zeros"),
        ("snowella",    "rare      | ~93% zeros"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for ci, (taxon, label) in enumerate(selected):
        raw  = df_clean[taxon].values.astype(float)
        logv = np.log(raw + eps)
        zlog = (logv - logv.mean()) / (logv.std() + 1e-10)

        # Row 0: raw counts
        axes[0, ci].hist(raw, bins=60, color="steelblue", alpha=0.8, edgecolor="none")
        axes[0, ci].set_title(f"{taxon}\n({label})", fontsize=8)
        axes[0, ci].text(
            0.97, 0.95,
            f"skew={stats.skew(raw):.1f}\nzeros={(raw == 0).mean():.0%}",
            transform=axes[0, ci].transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        axes[0, ci].grid(True, alpha=0.3)

        # Row 1: log-transformed
        axes[1, ci].hist(logv, bins=60, color="darkorange", alpha=0.8, edgecolor="none")
        axes[1, ci].text(
            0.97, 0.95, f"skew={stats.skew(logv):.1f}",
            transform=axes[1, ci].transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        axes[1, ci].grid(True, alpha=0.3)

        # Row 2: log + z-score vs N(0,1)
        axes[2, ci].hist(zlog, bins=60, color="seagreen", alpha=0.8,
                         edgecolor="none", density=True, label="log+zscore")
        x = np.linspace(zlog.min(), zlog.max(), 200)
        axes[2, ci].plot(x, stats.norm.pdf(x), "k--", lw=1.5, label="N(0,1)")
        axes[2, ci].text(
            0.97, 0.95, f"skew={stats.skew(zlog):.1f}",
            transform=axes[2, ci].transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        axes[2, ci].legend(fontsize=6)
        axes[2, ci].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Raw counts\n(organisms/μL)")
    axes[1, 0].set_ylabel("Log-transformed\nlog(v + ε)")
    axes[2, 0].set_ylabel("Log + z-score\nvs N(0,1)")
    fig.suptitle(
        "Marginal distributions: raw → log-transform → z-score\n"
        "All taxa are zero-inflated and right-skewed; log-transform substantially reduces skew",
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 5 — NaN block structure
# ---------------------------------------------------------------------------

def fig5_nan_structure(df_raw, taxa_cols, out_dir):
    nan_per_row = df_raw[taxa_cols].isna().sum(axis=1)
    row_sum     = df_raw[taxa_cols].sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    axes[0].scatter(df_raw["date"], nan_per_row, s=3, color="firebrick", alpha=0.6)
    axes[0].set_ylabel("NaN count per row")
    axes[0].set_title(
        "NaN structure — block outages (same taxa missing for entire months), not random scatter"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(df_raw["date"], row_sum, s=3, color="steelblue", alpha=0.5)
    axes[1].set_ylabel("Row sum (organisms/μL)")
    axes[1].set_title(
        "Row sum for context — NaN blocks do not coincide with low-abundance periods"
    )
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig5_nan_structure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path  = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data   : {data_path}")
    print(f"Output : {output_dir}")
    print()

    df_clean, df_nan, df_raw, taxa_cols = load_clean(data_path)

    print("\n[Fig 1] Row sum time series")
    fig1_rowsum_timeseries(df_clean, taxa_cols, output_dir)

    print("\n[Fig 2] Lomb-Scargle periodogram")
    fig2_lombscargle(df_clean, taxa_cols, output_dir)

    print("\n[Fig 3] Annual statistics + seasonal shape")
    fig3_annual_seasonal(df_clean, taxa_cols, output_dir)

    print("\n[Fig 4] Marginal distributions")
    fig4_distributions(df_clean, taxa_cols, output_dir)

    print("\n[Fig 5] NaN block structure")
    fig5_nan_structure(df_raw, taxa_cols, output_dir)

    print("\nDone.")