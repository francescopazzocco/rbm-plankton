"""
hidden_cross_model.py - Cross-model comparison: NB-RBM vs BB-median at L=6.

Three analyses:
  1. Pairwise Pearson correlation between NB and BB-median hidden unit activation
     timeseries - identifies which units encode the same seasonal signal.
  2. NB binary pattern frequency - 6-unit activation vector thresholded at 0.5;
     counts distinct patterns (effective state usage out of 2^6=64).
  3. Seasonal profiles - mean activation per month per unit for both models.

Outputs (CSVs) in results/figures/hidden/:
  cross_model_correlation.csv    - 6x6 Pearson matrix
  cross_model_matched_pairs.csv  - best NB<->BB match per unit
  nb_pattern_frequency.csv       - binary pattern counts
  seasonal_profiles_nb.csv       - mean activation by month
  seasonal_profiles_bb.csv       - mean activation by month

Outputs (plots) in results/figures/hidden/:
  cross_model_correlation.png
  nb_pattern_frequency.png
  seasonal_profiles.png
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.io import best_seed_dir, METRIC_COL
from models.visualization import plot_correlation, plot_pattern_frequency, plot_seasonal_profiles

RESULTS_DIR = Path(__file__).parent.parent / "results" / "training_runs"
OUT_DIR     = Path(__file__).parent.parent / "results" / "figures" / "hidden"
TARGET_L    = 6


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


def nb_pattern_frequency(nb: pd.DataFrame) -> pd.DataFrame:
    binary = (nb >= 0.5).astype(int)
    patterns = binary.apply(lambda row: "".join(row.astype(str)), axis=1)
    counts = patterns.value_counts()
    total  = len(patterns)
    df = pd.DataFrame({
        "pattern":  counts.index,
        "n_days":   counts.values,
        "fraction": (counts.values / total).round(4),
    })
    df["n_units_on"] = df["pattern"].apply(lambda p: p.count("1"))
    return df.reset_index(drop=True)


def seasonal_profile(act: pd.DataFrame) -> pd.DataFrame:
    act = act.copy()
    act["month"] = act.index.month
    return act.groupby("month").mean().round(4)


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
    print("\nBest NB<->BB matches:")
    print(pairs.to_string(index=False))

    plot_correlation(corr, OUT_DIR, target_l=TARGET_L)

    # 2. NB pattern frequency
    freq = nb_pattern_frequency(nb)
    freq.to_csv(OUT_DIR / "nb_pattern_frequency.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/nb_pattern_frequency.csv")
    print(f"Distinct NB patterns used: {len(freq)} / 64")
    print(freq.head(10).to_string(index=False))

    plot_pattern_frequency(freq, OUT_DIR, target_l=TARGET_L)

    # 3. Seasonal profiles
    nb_prof = seasonal_profile(nb)
    bb_prof = seasonal_profile(bb)
    nb_prof.to_csv(OUT_DIR / "seasonal_profiles_nb.csv")
    bb_prof.to_csv(OUT_DIR / "seasonal_profiles_bb.csv")
    print(f"\nSaved seasonal profiles.")

    plot_seasonal_profiles(nb_prof, bb_prof, OUT_DIR, target_l=TARGET_L)


if __name__ == "__main__":
    main()
