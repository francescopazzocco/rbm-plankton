"""
hidden_cross_model.py — Cross-model comparison: NB-RBM vs BB-median at L=6.

Two analyses:
  1. Pairwise Pearson correlation between NB and BB-median unit activation
     timeseries over shared dates — identifies which units encode the same
     seasonal signal regardless of dominant-state assignment.
  2. NB binary pattern frequency — each day's 6-unit activation vector is
     thresholded at 0.5 to a 6-bit string; counts how many distinct patterns
     appear and how often (measures effective state usage out of 2^6=64).
  3. Seasonal profiles — mean activation per month per unit for both models,
     to show what each unit encodes ecologically.

Outputs (CSVs):
  results/hidden/cross_model_correlation.csv    — 6×6 Pearson matrix
  results/hidden/cross_model_matched_pairs.csv  — best NB↔BB match per unit
  results/hidden/nb_pattern_frequency.csv       — binary pattern counts
  results/hidden/seasonal_profiles_nb.csv       — mean activation by month
  results/hidden/seasonal_profiles_bb.csv       — mean activation by month

Outputs (plots):
  results/hidden/cross_model_correlation.png
  results/hidden/nb_pattern_frequency.png
  results/hidden/seasonal_profiles.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results" / "multiseed_pcd"
OUT_DIR     = Path(__file__).parent.parent / "results" / "hidden"
TARGET_L    = 6

METRIC_COL = {
    "nb":               "val_nll",
    "bernoulli_median": "val_pll",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def best_seed_dir(family_l_dir: Path, metric_col: str) -> Path | None:
    best_val, best_dir = float("inf"), None
    for seed_dir in sorted(family_l_dir.glob("seed_*")):
        csv = seed_dir / "rbm_training_curves.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if metric_col not in df.columns:
            continue
        series = df[metric_col].dropna()
        if series.empty:
            continue
        v = series.iloc[-1]
        if v < best_val:
            best_val, best_dir = v, seed_dir
    return best_dir


def load_activations(family: str) -> pd.DataFrame:
    family_l_dir = RESULTS_DIR / f"{family}_L{TARGET_L}"
    seed_dir = best_seed_dir(family_l_dir, METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(f"No converged seed for {family} L={TARGET_L}")
    df = pd.read_csv(seed_dir / "rbm_hidden_activations.csv",
                     index_col="date", parse_dates=True)
    print(f"Loaded {family} L={TARGET_L} from {seed_dir.name}  "
          f"({len(df)} days, {df.shape[1]} units)")
    return df


# ── 1. Pairwise correlation ───────────────────────────────────────────────────

def compute_correlation(nb: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    shared = nb.index.intersection(bb.index)
    nb_s, bb_s = nb.loc[shared], bb.loc[shared]
    corr = pd.DataFrame(index=nb_s.columns, columns=bb_s.columns, dtype=float)
    for nb_col in nb_s.columns:
        for bb_col in bb_s.columns:
            corr.loc[nb_col, bb_col] = nb_s[nb_col].corr(bb_s[bb_col])
    return corr


def matched_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nb_unit in corr.index:
        best_bb = corr.loc[nb_unit].abs().idxmax()
        r = corr.loc[nb_unit, best_bb]
        rows.append({"nb_unit": nb_unit, "best_bb_match": best_bb,
                     "pearson_r": round(r, 4), "abs_r": round(abs(r), 4)})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def plot_correlation(corr: pd.DataFrame, out_dir: Path):
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
    ax.set_title(f"NB vs BB-median activation correlation  (L={TARGET_L})",
                 fontsize=11)
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


# ── 2. NB binary pattern frequency ───────────────────────────────────────────

def nb_pattern_frequency(nb: pd.DataFrame) -> pd.DataFrame:
    binary = (nb >= 0.5).astype(int)
    patterns = binary.apply(lambda row: "".join(row.astype(str)), axis=1)
    counts = patterns.value_counts()
    total = len(patterns)
    df = pd.DataFrame({
        "pattern": counts.index,
        "n_days":  counts.values,
        "fraction": (counts.values / total).round(4),
    })
    df["n_units_on"] = df["pattern"].apply(lambda p: p.count("1"))
    return df.reset_index(drop=True)


def plot_pattern_frequency(freq: pd.DataFrame, out_dir: Path):
    top = freq.head(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(top)), top["fraction"],
                  color=plt.cm.tab10(top["n_units_on"] / 6))
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["pattern"], fontsize=8, rotation=45, ha="right",
                       fontfamily="monospace")
    ax.set_ylabel("fraction of days")
    ax.set_title(f"NB L={TARGET_L} — most common 6-bit activation patterns "
                 f"(threshold=0.5, top 15 of {len(freq)} distinct)")
    ax.set_xlabel("binary pattern  (h0…h5, 1=ON)")
    for bar, row in zip(bars, top.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{row.n_days}d", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out = out_dir / "nb_pattern_frequency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── 3. Seasonal profiles ──────────────────────────────────────────────────────

def seasonal_profile(act: pd.DataFrame) -> pd.DataFrame:
    act = act.copy()
    act["month"] = act.index.month
    return act.groupby("month").mean().round(4)


def plot_seasonal_profiles(nb_prof: pd.DataFrame, bb_prof: pd.DataFrame,
                           out_dir: Path):
    n_units = nb_prof.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    months = range(1, 13)
    cmap = plt.cm.tab10

    for ax, prof, title in zip(axes,
                                [nb_prof, bb_prof],
                                [f"NB-RBM  (L={TARGET_L})",
                                 f"BB-median  (L={TARGET_L})"]):
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
    axes[-1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"],
                              fontsize=8)
    axes[-1].set_xlabel("month")
    fig.suptitle("Seasonal activation profiles — NB vs BB-median", fontsize=11)
    fig.tight_layout()
    out = out_dir / "seasonal_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nb = load_activations("nb")
    bb = load_activations("bernoulli_median")

    shared = nb.index.intersection(bb.index)
    print(f"Shared dates: {len(shared)}")

    # 1. Correlation
    corr = compute_correlation(nb, bb)
    corr.to_csv(OUT_DIR / "cross_model_correlation.csv")
    print(f"Saved: {OUT_DIR}/cross_model_correlation.csv")

    pairs = matched_pairs(corr)
    pairs.to_csv(OUT_DIR / "cross_model_matched_pairs.csv", index=False)
    print(f"Saved: {OUT_DIR}/cross_model_matched_pairs.csv")
    print("\nBest NB↔BB matches:")
    print(pairs.to_string(index=False))

    plot_correlation(corr, OUT_DIR)

    # 2. NB pattern frequency
    freq = nb_pattern_frequency(nb)
    freq.to_csv(OUT_DIR / "nb_pattern_frequency.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/nb_pattern_frequency.csv")
    print(f"Distinct NB patterns used: {len(freq)} / 64")
    print(freq.head(10).to_string(index=False))

    # 3. Seasonal profiles
    nb_prof = seasonal_profile(nb)
    bb_prof = seasonal_profile(bb)
    nb_prof.to_csv(OUT_DIR / "seasonal_profiles_nb.csv")
    bb_prof.to_csv(OUT_DIR / "seasonal_profiles_bb.csv")
    print(f"\nSaved seasonal profiles.")
    plot_seasonal_profiles(nb_prof, bb_prof, OUT_DIR)


if __name__ == "__main__":
    main()
