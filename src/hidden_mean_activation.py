"""
hidden_mean_activation.py - Mean hidden unit activation across L values.

For each model family and L, computes the mean activation of each hidden unit
over all samples. Identifies always-on (bias absorber) and always-off units.

Output: results/figures/hidden/mean_activation_{family}.png
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.io import best_seed_dir, METRIC_COL
from models.visualization import ABSORBER_HI, ABSORBER_LO, mean_activations, plot_family

RESULTS_DIR = Path(__file__).parent.parent / "results" / "training_runs"
OUT_DIR     = Path(__file__).parent.parent / "results" / "figures" / "hidden"

FAMILIES = ["bernoulli_median", "bernoulli_zero", "nb"]


def discover_runs(results_dir: Path) -> dict[str, dict[int, Path]]:
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, Path]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        metric_col = METRIC_COL.get(family)
        if metric_col is None:
            continue
        seed_dir = best_seed_dir(d, metric_col)
        if seed_dir is None:
            continue
        csv = seed_dir / "rbm_hidden_activations.csv"
        if csv.exists():
            runs.setdefault(family, {})[l_val] = csv
    return runs


def save_summary_csv(runs: dict, out_dir: Path):
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
    out = out_dir / "mean_activation_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    for family in FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        plot_family(family, runs[family], OUT_DIR)

    save_summary_csv(runs, OUT_DIR)


if __name__ == "__main__":
    main()
