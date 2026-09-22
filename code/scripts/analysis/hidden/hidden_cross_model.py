"""
hidden_cross_model.py - Cross-model comparison: NB-RBM vs BB-median at L=6.

Three analyses:
  1. Pairwise Pearson correlation between NB and BB-median hidden unit activation
     timeseries - identifies which units encode the same seasonal signal.
  2. NB binary pattern frequency - 6-unit activation vector thresholded at 0.5;
     counts distinct patterns (effective state usage out of 2^6=64).
  3. Seasonal profiles - mean activation per month per unit for both models.

Outputs (CSVs) in results/tables/hidden/:
  cross_model_correlation.csv    - 6x6 Pearson matrix
  cross_model_matched_pairs.csv  - best NB<->BB match per unit
  nb_pattern_frequency.csv       - binary pattern counts
  seasonal_profiles_nb.csv       - mean activation by month
  seasonal_profiles_bb.csv       - mean activation by month

Outputs (plots) in results/02_model_analysis/hidden/:
  cross_model_correlation_L{L}.png
  nb_pattern_frequency_L{L}.png
  seasonal_profiles_L{L}.png

Usage:
    python code/scripts/analysis/hidden/hidden_cross_model.py [--split chrono|shuffled] [--L 6]
"""

import argparse
from pathlib import Path

import pandas as pd
from models.io import (
    CHRONO,
    METRIC_COL,
    SPLITS,
    best_seed_dir,
    load_hidden_activations,
    run_dir,
    split_out_dir,
)
from models.paths import DIAGNOSTIC_ROOT, RUNS_ROOT
from models.visualization import (
    hidden_binary,
    pattern_frequency,
    plot_correlation,
    plot_pattern_frequency,
    plot_seasonal_profiles,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of hidden activations: NB-RBM vs BB-median.")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Which split strategy's runs to compare (default: chrono)")
    parser.add_argument("--L", type=int, default=6,
                        help="Hidden unit count to compare (default: 6)")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT,
                        help="Directory holding the {family}_L{n} run directories")
    return parser.parse_args(argv)


def load_activations(family: str, n_hidden: int, split: str,
                     runs_root: Path) -> pd.DataFrame:
    family_l_dir = run_dir(family, n_hidden, split, runs_root)
    seed_dir = best_seed_dir(family_l_dir, METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(f"No converged seed for {family} L={n_hidden} ({split})")
    df = load_hidden_activations(seed_dir / "rbm_hidden_activations.csv")
    print(f"Loaded {family} L={n_hidden} from {seed_dir.name}  "
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


def seasonal_profile(act: pd.DataFrame) -> pd.DataFrame:
    act = act.copy()
    act["month"] = act.index.month
    return act.groupby("month").mean().round(4)


def main():
    args = parse_args()
    csv_dir = split_out_dir(DIAGNOSTIC_ROOT / "tables" / "hidden", args.split)
    fig_dir = split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis" / "hidden", args.split)
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    nb = load_activations("nb", args.L, args.split, args.runs_root)
    bb = load_activations("bernoulli_median", args.L, args.split, args.runs_root)

    shared = nb.index.intersection(bb.index)
    print(f"Shared dates: {len(shared)}")

    # 1. Correlation
    corr = compute_correlation(nb, bb)
    corr.to_csv(csv_dir / "cross_model_correlation.csv")
    print(f"Saved: {csv_dir}/cross_model_correlation.csv")

    pairs = matched_pairs(corr)
    pairs.to_csv(csv_dir / "cross_model_matched_pairs.csv", index=False)
    print(f"Saved: {csv_dir}/cross_model_matched_pairs.csv")
    print("\nBest NB<->BB matches:")
    print(pairs.to_string(index=False))

    plot_correlation(corr, fig_dir, target_l=args.L)

    # 2. NB pattern frequency
    freq = pattern_frequency(hidden_binary(nb))
    freq.to_csv(csv_dir / "nb_pattern_frequency.csv", index=False)
    print(f"\nSaved: {csv_dir}/nb_pattern_frequency.csv")
    print(f"Distinct NB patterns used: {len(freq)} / {2 ** args.L}")
    print(freq.head(10).to_string(index=False))

    plot_pattern_frequency(freq, fig_dir, target_l=args.L)

    # 3. Seasonal profiles
    nb_prof = seasonal_profile(nb)
    bb_prof = seasonal_profile(bb)
    nb_prof.to_csv(csv_dir / "seasonal_profiles_nb.csv")
    bb_prof.to_csv(csv_dir / "seasonal_profiles_bb.csv")
    print("\nSaved seasonal profiles.")

    plot_seasonal_profiles(nb_prof, bb_prof, fig_dir, target_l=args.L)


if __name__ == "__main__":
    main()
