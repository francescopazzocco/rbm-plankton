"""
hidden_coactivation.py - Community state analysis via weight profiles and temporal assignment.

Two figures per model family:
  1. Weight profiles: species x hidden unit heatmap (species sorted by dominant unit)
  2. Dominant state timeline: each date assigned to its argmax hidden unit, one row per L

Output: results/figures/hidden/weight_profiles_{family}.png
        results/figures/hidden/state_timeline_{family}.png
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models.io import best_seed_dir, METRIC_COL
from models.visualization import (
    load_activations, dominant_state,
    plot_weight_profiles, plot_state_timeline,
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "training_runs"
OUT_DIR     = Path(__file__).parent.parent / "results" / "figures" / "hidden"

FAMILIES = ["bernoulli_median", "bernoulli_zero", "nb"]


def discover_runs(results_dir: Path) -> dict[str, dict[int, dict[str, Path]]]:
    """Return {family: {L: {activations, weights}}} using the best seed per (family, L)."""
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, dict[str, Path]]] = {}
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
        act_csv = seed_dir / "rbm_hidden_activations.csv"
        w_csv   = seed_dir / "rbm_weights.csv"
        if act_csv.exists() and w_csv.exists():
            runs.setdefault(family, {})[l_val] = {
                "activations": act_csv,
                "weights":     w_csv,
            }
    return runs


def save_state_frequency(runs: dict, out_dir: Path):
    """State frequency table: fraction of days each unit is dominant per (family, L)."""
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
    df = pd.DataFrame(rows)
    out = out_dir / "state_frequency.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


def save_dominant_state_l6(runs: dict, out_dir: Path, target_l: int = 6):
    """Per-date dominant state for each family at target_l - for cross-model comparison."""
    frames = []
    for family, family_runs in runs.items():
        if target_l not in family_runs:
            continue
        act   = load_activations(family_runs[target_l]["activations"])
        state = dominant_state(act).rename(family)
        frames.append(state)
    if not frames:
        return
    df = pd.concat(frames, axis=1)
    df.index.name = "date"
    out = out_dir / f"dominant_state_L{target_l}.csv"
    df.to_csv(out)
    print(f"Saved: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    for family in FAMILIES:
        if family not in runs:
            print(f"No runs found for {family}, skipping.")
            continue
        print(f"\n-- {family} --")
        plot_weight_profiles(family, runs[family], OUT_DIR)
        plot_state_timeline(family, runs[family], OUT_DIR)

    save_state_frequency(runs, OUT_DIR)
    save_dominant_state_l6(runs, OUT_DIR)


if __name__ == "__main__":
    main()
