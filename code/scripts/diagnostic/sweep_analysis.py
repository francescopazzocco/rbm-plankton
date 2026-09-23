"""
sweep_analysis.py - NLL/PLL vs L sweep analysis across all trained models.

Reads artifacts/models/{family}/{split}/L{n}/seed_*/rbm_training_curves.csv and
produces figures in:
  diagnostic_outputs/04_model_selection/{split}/ — final val metric vs L per model family
  diagnostic_outputs/diagnostics/training_curves/all_families_by_L/{split}/ — val metric vs epoch, all families
  diagnostic_outputs/diagnostics/nb_zinb_parameters/{split}/ — NB/ZINB val NLL + theta (+ pi) trajectories by L

Each split writes to its own chrono/ or shuffled/ subdirectory, so the two
splits cannot overwrite each other's figures.

Usage:
    python code/scripts/diagnostic/sweep_analysis.py [--split chrono|shuffled]
"""

import argparse
from pathlib import Path

import pandas as pd

from models.io import CHRONO, SPLITS, discover_model_dirs, split_out_dir
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.visualization import (
    FAMILY_META, aggregate_curves,
    plot_final_metric, plot_final_metric_overview, plot_final_metric_individual,
    plot_sweep_curves, plot_nb_diagnostics, plot_zinb_diagnostics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLL/PLL vs L sweep analysis.")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO,
                        help="Which split strategy's runs to analyse (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT,
                        help="Directory holding artifacts/models/{family}/{split}/L{n} directories")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    curves_dir = split_out_dir(DIAGNOSTIC_ROOT / "diagnostics" / "training_curves" / "all_families_by_L", args.split)
    params_dir = split_out_dir(DIAGNOSTIC_ROOT / "diagnostics" / "nb_zinb_parameters", args.split)
    metric_dir = split_out_dir(DIAGNOSTIC_ROOT / "04_model_selection", args.split)
    curves_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    all_dirs = discover_model_dirs(args.models_root, args.split)
    runs: dict[str, dict[int, list[Path]]] = {}
    for family, ls in all_dirs.items():
        for l_val, seed_dirs in ls.items():
            csvs = sorted(sd / "rbm_training_curves.csv" for sd in seed_dirs
                          if (sd / "rbm_training_curves.csv").exists())
            if csvs:
                runs.setdefault(family, {})[l_val] = csvs

    print("Discovered runs:")
    for family, ls in sorted(runs.items()):
        details = ", ".join(f"L{l}({len(ls[l])} seeds)" for l in sorted(ls.keys()))
        print(f"  {family}: {details}")

    print("\n-- Relative improvement per L step --")
    print(f"{'family':<22} {'L->L+1':<10} {'val metric':<18} {'delta abs':<10} {'delta %'}")
    print("-" * 75)
    for family, meta in FAMILY_META.items():
        col    = meta["col"]
        better = meta["better"]
        family_runs = runs.get(family, {})
        pairs: list[tuple[int, float, float]] = []
        for l_val in sorted(family_runs):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is not None:
                mean_curve, std_curve = agg
                pairs.append((l_val, mean_curve.iloc[-1], std_curve.iloc[-1]))
        for i in range(len(pairs) - 1):
            l_a, v_a, s_a = pairs[i]
            l_b, v_b, s_b = pairs[i + 1]
            delta  = v_b - v_a
            rel    = delta / abs(v_a) * 100
            marker = ""
            if better == "higher" and delta > 0:
                marker = "↑"
            elif better == "lower" and delta < 0:
                marker = "↓"
            print(f"{family:<22} {l_a}->{l_b:<8} {v_a:.4f}±{s_a:.4f}  {delta:+.4f}   {rel:+.2f}% {marker}")
        print()

    plot_final_metric(runs, metric_dir)
    plot_final_metric_overview(runs, metric_dir)
    plot_final_metric_individual(runs, metric_dir)
    plot_sweep_curves(runs, curves_dir)
    plot_nb_diagnostics(runs, params_dir)
    plot_zinb_diagnostics(runs, params_dir)


if __name__ == "__main__":
    main()
