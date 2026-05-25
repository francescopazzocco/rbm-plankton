"""
hidden_mean_activation.py - Mean hidden unit activation across L values.

For each model family and L, computes the mean activation of each hidden unit
over all samples. Identifies always-on (bias absorber) and always-off units.

Output: results/02_model_analysis/mean_activation_{family}.png
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models._constants import ALL_FAMILIES
from models.io import best_seed_dir, discover_run_dirs, METRIC_COL
from models.visualization import ABSORBER_HI, ABSORBER_LO, mean_activations, plot_family

RESULTS_DIR = Path(__file__).parent.parent.parent / "trained_models"
SUFFIX = ""
OUT_DIR = Path(__file__).parent.parent.parent / "results" / "02_model_analysis" / ("shuffled" if SUFFIX else "")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dirs = discover_run_dirs(RESULTS_DIR, SUFFIX)

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
        plot_family(family, runs[family], OUT_DIR)

    suffix = "_shuffled" if SUFFIX else ""
    rows = []
    for family, family_runs in runs.items():
        for l_val, csv in family_runs.items():
            means = mean_activations(csv)
            for unit, v in means.items():
                flag = ("absorber_hi" if v >= ABSORBER_HI
                        else "absorber_lo" if v <= ABSORBER_LO
                        else "active")
                rows.append({"family": family, "L": l_val,
                             "unit": unit, "mean_activation": round(v, 4),
                             "flag": flag})
    df = pd.DataFrame(rows)
    out = OUT_DIR / f"mean_activation_summary{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
