"""
sweep_analysis.py - NLL/PLL vs L sweep analysis across all trained models.

Reads trained_models/{family}_L{n}/seed_*/rbm_training_curves.csv and
produces figures in:
  diagnostic_outputs/04_model_selection/ — final val metric vs L per model family
  diagnostic_outputs/diagnostics/sweep{SUFFIX}/ — training curves, NB/ZINB diagnostics
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models.io import discover_run_dirs
from models.visualization import (
    FAMILY_META, aggregate_curves,
    plot_final_metric, plot_sweep_curves, plot_nb_diagnostics, plot_zinb_diagnostics,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "trained_models"
SHUFFLED = False
SUFFIX = "_shuffled" if SHUFFLED else ""
DIAG_DIR = Path(__file__).parent.parent.parent / "diagnostic_outputs" / "diagnostics" / f"sweep{SUFFIX}"
METRIC_DIR = Path(__file__).parent.parent.parent / "diagnostic_outputs" / "04_model_selection"


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    all_dirs = discover_run_dirs(RESULTS_DIR, SUFFIX)
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

    plot_final_metric(runs, METRIC_DIR)
    plot_sweep_curves(runs, DIAG_DIR)
    plot_nb_diagnostics(runs, DIAG_DIR)
    plot_zinb_diagnostics(runs, DIAG_DIR)


if __name__ == "__main__":
    main()
