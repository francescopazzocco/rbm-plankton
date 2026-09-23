"""
hidden_cross_model.py - Cross-model comparison: NB-family RBM vs BB-median.

--family selects which NB hidden-unit variant to compare (nb / nb_relu /
nb_sigmoid / nb_softmax). Default is "nb" (Bernoulli hidden) for
backward-compatible output paths, NOT the model DECISION_LOG LOG-021
recommends (nb_sigmoid) -- pass --family nb_sigmoid explicitly for that.

Three analyses:
  1. Pairwise Pearson correlation between the chosen NB variant and BB-median
     hidden unit activation timeseries - identifies which units encode the
     same seasonal signal.
  2. Binary pattern frequency - L-unit activation vector thresholded at 0.5;
     counts distinct patterns (effective state usage out of 2^L).
  3. Seasonal profiles - mean activation per month per unit for both models.

Outputs (CSVs) in results/tables/hidden/{split}/, unsuffixed for the default
family="nb" (nb_sigmoid etc. get a "_sigmoid" suffix instead of overwriting
the nb= Bernoulli baseline's tables -- see archetype_rbm_comparison.py, which
hardcodes the unsuffixed nb paths):
  cross_model_correlation[_<profile>].csv    - LxL Pearson matrix
  cross_model_matched_pairs[_<profile>].csv  - best NB<->BB match per unit
  nb_pattern_frequency[_<profile>].csv       - binary pattern counts
  seasonal_profiles_nb[_<profile>].csv       - mean activation by month
  seasonal_profiles_bb.csv                   - mean activation by month (family-independent)

Outputs (plots), one kind-subfolder per analysis:
  results/02_model_analysis/hidden/cross_model_correlation/{split}/[<profile>_]L{L}.png
  results/02_model_analysis/hidden/nb_pattern_frequency/{split}/[<profile>_]L{L}.png
  results/02_model_analysis/hidden/seasonal_profiles/{split}/[<profile>_]L{L}.png

Usage:
    python code/scripts/analysis/hidden/hidden_cross_model.py \\
        [--split chrono|shuffled] [--L 6] [--family nb|nb_relu|nb_sigmoid|nb_softmax]
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
    model_dir,
    split_out_dir,
)
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
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
    parser.add_argument("--family", default="nb",
                        choices=["nb", "nb_relu", "nb_sigmoid", "nb_softmax"],
                        help="NB hidden-unit variant to compare against BB-median "
                             "(default: nb, i.e. Bernoulli hidden -- NOT the "
                             "LOG-021-recommended variant; pass nb_sigmoid for that)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="Directory holding artifacts/models/{family}/{split}/L{n} directories")
    return parser.parse_args(argv)


def load_activations(family: str, n_hidden: int, split: str,
                     models_root: Path) -> pd.DataFrame:
    family_l_dir = model_dir(family, n_hidden, split, models_root)
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
    hidden_base = DIAGNOSTIC_ROOT / "02_model_analysis" / "hidden"
    csv_dir = split_out_dir(DIAGNOSTIC_ROOT / "tables" / "hidden", args.split)
    correlation_dir  = split_out_dir(hidden_base / "cross_model_correlation", args.split)
    pattern_freq_dir = split_out_dir(hidden_base / "nb_pattern_frequency", args.split)
    seasonal_dir     = split_out_dir(hidden_base / "seasonal_profiles", args.split)
    for d in (csv_dir, correlation_dir, pattern_freq_dir, seasonal_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Output filenames stay unsuffixed for the default family="nb" (Bernoulli
    # hidden) so existing chrono/shuffled L6 outputs -- and
    # archetype_rbm_comparison.py's hardcoded paths to seasonal_profiles_nb.csv
    # -- are untouched. A non-default family (e.g. nb_sigmoid, LOG-021's
    # recommended NB hidden-unit type) gets its own suffixed files instead of
    # silently overwriting the nb= Bernoulli baseline's tables.
    suffix = "" if args.family == "nb" else "_" + args.family[len("nb") + 1:]

    nb = load_activations(args.family, args.L, args.split, args.models_root)
    bb = load_activations("bernoulli_median", args.L, args.split, args.models_root)

    shared = nb.index.intersection(bb.index)
    print(f"Shared dates: {len(shared)}")

    # 1. Correlation
    corr = compute_correlation(nb, bb)
    corr.to_csv(csv_dir / f"cross_model_correlation{suffix}.csv")
    print(f"Saved: {csv_dir}/cross_model_correlation{suffix}.csv")

    pairs = matched_pairs(corr)
    pairs.to_csv(csv_dir / f"cross_model_matched_pairs{suffix}.csv", index=False)
    print(f"Saved: {csv_dir}/cross_model_matched_pairs{suffix}.csv")
    print(f"\nBest {args.family}<->BB matches:")
    print(pairs.to_string(index=False))

    plot_correlation(corr, correlation_dir, target_l=args.L, family=args.family)

    # 2. NB pattern frequency
    freq = pattern_frequency(hidden_binary(nb))
    freq.to_csv(csv_dir / f"nb_pattern_frequency{suffix}.csv", index=False)
    print(f"\nSaved: {csv_dir}/nb_pattern_frequency{suffix}.csv")
    print(f"Distinct {args.family} patterns used: {len(freq)} / {2 ** args.L}")
    print(freq.head(10).to_string(index=False))

    plot_pattern_frequency(freq, pattern_freq_dir, target_l=args.L, family=args.family)

    # 3. Seasonal profiles
    nb_prof = seasonal_profile(nb)
    bb_prof = seasonal_profile(bb)
    nb_prof.to_csv(csv_dir / f"seasonal_profiles_nb{suffix}.csv")
    bb_prof.to_csv(csv_dir / "seasonal_profiles_bb.csv")
    print("\nSaved seasonal profiles.")

    plot_seasonal_profiles(nb_prof, bb_prof, seasonal_dir, target_l=args.L, family=args.family)


if __name__ == "__main__":
    main()
