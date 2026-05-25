"""
hidden_coactivation.py - Community state analysis via weight profiles and temporal assignment.

Two figures per model family:
  1. Weight profiles: species x hidden unit heatmap (species sorted by dominant unit)
  2. Dominant state timeline: each date assigned to its argmax hidden unit, one row per L

Output: results/02_model_analysis/weight_profiles_{family}.png
        results/02_model_analysis/state_timeline_{family}.png
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models._constants import ALL_FAMILIES
from models.io import best_seed_dir, discover_run_dirs, METRIC_COL
from models.visualization import (
    load_activations, dominant_state,
    plot_weight_profiles, plot_state_timeline,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "trained_models"
SUFFIX = ""
OUT_DIR = Path(__file__).parent.parent.parent / "results" / "02_model_analysis" / ("shuffled" if SUFFIX else "")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_dirs = discover_run_dirs(RESULTS_DIR, SUFFIX)

    runs: dict[str, dict[int, dict[str, Path]]] = {}
    for family in ALL_FAMILIES:
        ls = all_dirs.get(family, {})
        metric_col = METRIC_COL.get(family)
        if metric_col is None:
            continue
        for l_val, seed_dirs in ls.items():
            seed_dir = best_seed_dir(seed_dirs[0].parent, metric_col)
            if seed_dir is None:
                continue
            act_csv = seed_dir / "rbm_hidden_activations.csv"
            w_csv   = seed_dir / "rbm_weights.csv"
            if act_csv.exists() and w_csv.exists():
                runs.setdefault(family, {})[l_val] = {
                    "activations": act_csv,
                    "weights":     w_csv,
                }

    for family in ALL_FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        print(f"\n-- {family} --")
        plot_weight_profiles(family, runs[family], OUT_DIR)
        plot_state_timeline(family, runs[family], OUT_DIR)

    suffix = "_shuffled" if SUFFIX else ""

    rows = []
    for family, family_runs in runs.items():
        for l_val, paths in family_runs.items():
            act   = load_activations(paths["activations"])
            state = dominant_state(act)
            counts = state.value_counts().sort_index()
            total  = len(state)
            for unit in range(l_val):
                n = counts.get(unit, 0)
                rows.append({"family": family, "L": l_val, "unit": f"h{unit}",
                             "n_days": int(n), "fraction": round(n / total, 4)})
    df_state = pd.DataFrame(rows)
    out = OUT_DIR / f"state_frequency{suffix}.csv"
    df_state.to_csv(out, index=False)
    print(f"Saved: {out}")

    frames = []
    for family, family_runs in runs.items():
        if 6 not in family_runs:
            continue
        act   = load_activations(family_runs[6]["activations"])
        state = dominant_state(act).rename(family)
        frames.append(state)
    if frames:
        df_dom = pd.concat(frames, axis=1)
        df_dom.index.name = "date"
        out = OUT_DIR / f"dominant_state_L6{suffix}.csv"
        df_dom.to_csv(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
