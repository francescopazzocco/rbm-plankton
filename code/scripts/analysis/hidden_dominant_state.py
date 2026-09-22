"""
hidden_dominant_state.py - Community state analysis: each date assigned to its
argmax hidden unit, plus the weight profiles needed to read those states.

Named for what it computes.  It was called hidden_coactivation.py, but no
coactivation quantity appears anywhere in it.

Two figures per model family:
  1. Weight profiles: species x hidden unit heatmap (species sorted by dominant unit)
  2. Dominant state timeline: each date assigned to its argmax hidden unit, one row per L

Output: results/02_model_analysis/weight_profiles_{family}.png
        results/02_model_analysis/state_timeline_{family}.png
        (shuffled-split runs under results/02_model_analysis/shuffled/)

Usage:
    python code/scripts/analysis/hidden_dominant_state.py [--split chrono|shuffled]
"""

import argparse
from pathlib import Path

import pandas as pd

from models._constants import ALL_FAMILIES
from models.io import (
    CHRONO, SPLITS, best_seed_dir, discover_run_dirs, METRIC_COL,
    split_out_dir, split_suffix,
)
from models.paths import DIAGNOSTIC_ROOT, RUNS_ROOT
from models.visualization import (
    load_activations, dominant_state,
    plot_weight_profiles, plot_state_timeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dominant hidden state per date, with supporting weight profiles.")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Which split strategy's runs to analyse (default: chrono)")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT,
                        help="Directory holding the {family}_L{n} run directories")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    out_dir = split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis", args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dirs = discover_run_dirs(args.runs_root, args.split)

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
        plot_weight_profiles(family, runs[family], out_dir)
        plot_state_timeline(family, runs[family], out_dir)

    suffix = split_suffix(args.split)

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
    out = out_dir / f"state_frequency{suffix}.csv"
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
        out = out_dir / f"dominant_state_L6{suffix}.csv"
        df_dom.to_csv(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
