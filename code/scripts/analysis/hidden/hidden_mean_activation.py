"""
hidden_mean_activation.py - Mean hidden unit activation across L values.

For each model family and L, computes the mean activation of each hidden unit
over all samples. Identifies always-on (bias absorber) and always-off units.

Output: results/02_model_analysis/hidden/mean_activation/{chrono,shuffled}/{family}.png
        results/02_model_analysis/hidden/mean_activation/{chrono,shuffled}/summary.csv

Usage:
    python code/scripts/analysis/hidden/hidden_mean_activation.py [--split chrono|shuffled]
"""

import argparse
from pathlib import Path

import pandas as pd
from models._constants import ALL_FAMILIES
from models.io import (
    CHRONO,
    METRIC_COL,
    SPLITS,
    best_seed_dir,
    discover_model_dirs,
    split_out_dir,
)
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import ABSORBER_HI, absorber_lo, mean_activations, plot_family


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean hidden unit activation across L values.")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Which split strategy's runs to analyse (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="Directory holding artifacts/models/{family}/{split}/L{n} directories")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out_dir = split_out_dir(
        DIAGNOSTIC_ROOT / "02_model_analysis" / "hidden" / "mean_activation", args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dirs = discover_model_dirs(args.models_root, args.split)

    runs: dict[str, dict[int, Path]] = {}
    for family in ALL_FAMILIES:
        ls = all_dirs.get(family, {})
        metric_col = METRIC_COL.get(family)
        if metric_col is None:
            continue
        for l_val, seed_dirs in ls.items():
            seed_dir = best_seed_dir(seed_dirs[0].parent, metric_col)
            if seed_dir is None:
                continue
            csv = seed_dir / "rbm_hidden_activations.csv"
            if csv.exists():
                runs.setdefault(family, {})[l_val] = csv

    for family in ALL_FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        plot_family(family, runs[family], out_dir)

    rows = []
    for family, family_runs in runs.items():
        for l_val, csv in family_runs.items():
            means = mean_activations(csv)
            for unit, v in means.items():
                flag = ("absorber_hi" if v >= ABSORBER_HI
                        else "absorber_lo" if v <= absorber_lo(family)
                        else "active")
                rows.append({"family": family, "L": l_val,
                             "unit": unit, "mean_activation": round(v, 4),
                             "flag": flag})
    df = pd.DataFrame(rows)
    out = out_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
