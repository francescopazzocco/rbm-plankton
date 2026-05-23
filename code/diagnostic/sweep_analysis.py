"""
sweep_analysis.py - NLL/PLL vs L sweep analysis across all trained models.

Reads trained_models/{family}_L{n}/seed_*/rbm_training_curves.csv and
produces figures in:
  diagnostic_outputs/04_model_selection/ — final val metric vs L per model family
  diagnostic_outputs/diagnostics/sweep{SUFFIX}/ — training curves, NB/ZINB diagnostics
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from models.visualization import (
    FAMILY_META, aggregate_curves,
    plot_final_metric, plot_sweep_curves, plot_nb_diagnostics, plot_zinb_diagnostics,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "trained_models"
SHUFFLED = False  # set True to analyse shuffled-split runs
SUFFIX = "_shuffled" if SHUFFLED else ""
DIAG_DIR = Path(__file__).parent.parent.parent / "diagnostic_outputs" / "diagnostics" / f"sweep{SUFFIX}"
METRIC_DIR = Path(__file__).parent.parent.parent / "diagnostic_outputs" / "04_model_selection"


def discover_runs(results_dir: Path) -> dict[str, dict[int, list[Path]]]:
    """Return {family: {L: [csv_paths]}} for all valid result directories."""
    pattern = re.compile(rf"^(.+)_L(\d+){re.escape(SUFFIX)}$")
    runs: dict[str, dict[int, list[Path]]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        csvs = list(d.glob("seed_*/rbm_training_curves.csv"))
        if csvs:
            runs.setdefault(family, {})[l_val] = sorted(csvs)
    return runs


def print_improvement_table(runs):
    """Print relative val metric improvement per L step for each family."""
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


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    print("Discovered runs:")
    for family, ls in sorted(runs.items()):
        details = ", ".join(f"L{l}({len(ls[l])} seeds)" for l in sorted(ls.keys()))
        print(f"  {family}: {details}")

    print_improvement_table(runs)
    plot_final_metric(runs, METRIC_DIR)
    plot_sweep_curves(runs, DIAG_DIR)
    plot_nb_diagnostics(runs, DIAG_DIR)
    plot_zinb_diagnostics(runs, DIAG_DIR)


if __name__ == "__main__":
    main()
