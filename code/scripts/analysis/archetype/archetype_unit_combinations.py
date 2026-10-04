"""archetype_unit_combinations.py - Archetypes against combinations of hidden units.

With Sigmoid (or Bernoulli) hidden units a day is a combination of active units,
so an archetype should be compared with the visible profile of a combination,
not of one unit alone. For one run (best seed by default) this script computes,
for every archetype:

  1. baseline: cosine with the profile of h = 0 (all units off);
  2. best single unit: highest cosine over the L one-hot vectors h = e_j
     (the comparison of distance_archetypes_rbm.py --metric cosine);
  3. best combination: highest cosine over all 2^L binary vectors h;
  4. the combination itself (bit string h0...h_{L-1}, as in the pattern figure);
  5. its frequency rank among the patterns the days actually use (1 = most
     frequent, same ranking as hidden_pattern_analysis.py);
  6. the share of the archetype's days (dominant archetype) whose pattern is
     that combination.

The profile of h is the NB mean mu_i(h) = exp(a_i + sum_j W_ij h_j), clamped as in
the model; cosine is computed on the taxa shared by the run and the archetypes.
Days are discretised with hidden_binary (threshold 0.5), the rule of the
pattern-coverage figure.

Outputs (diagnostic_outputs/02_model_analysis/archetype/unit_combinations/{split}/):
  {family}_L{L}_unit_combinations.csv
  and, with --tex-out, the LaTeX table used in the paper.

Usage:
    python code/scripts/analysis/archetype/archetype_unit_combinations.py \
        [--family nb_sigmoid] [--L 6] [--split shuffled] [--tex-out paper/tab_archetype_combinations.tex]
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from models._constants import ETA_CLAMP_MAX
from models.io import (
    METRIC_COL,
    SHUFFLED,
    SPLITS,
    best_seed_dir,
    load_hidden_activations,
    model_dir,
    split_out_dir,
)
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.paths import PROJECT_ROOT as ROOT
from models.visualization import hidden_binary, pattern_frequency, pattern_labels


def unit_profile(W: np.ndarray, a: np.ndarray, h: np.ndarray) -> np.ndarray:
    """NB mean of every taxon when the hidden layer is set to h."""
    return np.exp(np.clip(a + W @ h, None, ETA_CLAMP_MAX))


def cosine_to_archetypes(arch: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Cosine between each (row-normalised) archetype and one profile."""
    return arch @ (profile / np.linalg.norm(profile))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="nb_sigmoid")
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--split", choices=SPLITS, default=SHUFFLED)
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--archetypes", type=Path,
                        default=ROOT / "data" / "archetypes" / "archetypes_k5_profiles.csv")
    parser.add_argument("--timeseries", type=Path,
                        default=ROOT / "data" / "archetypes" / "archetypes_k5_timeseries.csv")
    parser.add_argument("--tex-out", type=Path, default=None,
                        help="Also write the LaTeX table to this path")
    args = parser.parse_args()

    seed_dir = best_seed_dir(model_dir(args.family, args.L, args.split, args.models_root),
                             METRIC_COL[args.family])
    weights = np.load(seed_dir / "weights.npz", allow_pickle=True)
    W, a, taxa = weights["W"], weights["a"], list(weights["taxa"])

    profiles = pd.read_csv(args.archetypes, index_col=0)
    shared = [t for t in profiles.columns if t in set(taxa)]
    keep = [taxa.index(t) for t in shared]
    W, a = W[keep], a[keep]
    arch = profiles[shared].values.astype(float)
    arch /= np.linalg.norm(arch, axis=1, keepdims=True)
    arch_names = list(profiles.index)

    # every binary vector h, with its bit string in the h0...h_{L-1} order of the figure
    combos = np.array(list(itertools.product([0, 1], repeat=args.L)), dtype=float)
    labels = ["".join(str(int(x)) for x in h) for h in combos]
    cos = np.array([cosine_to_archetypes(arch, unit_profile(W, a, h)) for h in combos])
    n_on = combos.sum(axis=1)
    single = np.where(n_on == 1)[0]

    # day patterns and dominant archetypes on the shared days
    binary = hidden_binary(load_hidden_activations(seed_dir / "rbm_hidden_activations.csv"))
    day_pattern = pattern_labels(binary)
    day_pattern.index = pd.to_datetime(day_pattern.index)
    weights_ts = pd.read_csv(args.timeseries, parse_dates=["date"]).set_index("date")
    dominant = pd.Series(weights_ts.values.argmax(axis=1), index=weights_ts.index)
    days = day_pattern.index.intersection(dominant.index)
    day_pattern, dominant = day_pattern.loc[days], dominant.loc[days]
    freq = pattern_frequency(binary)
    rank = {p: i + 1 for i, p in enumerate(freq["pattern"])}

    rows = []
    for k, name in enumerate(arch_names):
        best_single = single[cos[single, k].argmax()]
        best = cos[:, k].argmax()
        own_days = day_pattern[dominant == k]
        rows.append({
            "archetype": name,
            "baseline_cos": cos[n_on == 0, k][0],
            "single_unit": f"h{int(combos[best_single].argmax())}",
            "single_cos": cos[best_single, k],
            "combination": labels[best],
            "combination_cos": cos[best, k],
            "frequency_rank": rank.get(labels[best], np.nan),
            "n_patterns_used": len(freq),
            "archetype_days": len(own_days),
            "share_of_archetype_days": (own_days == labels[best]).mean(),
            "days_of_other_archetypes": int((day_pattern[dominant != k] == labels[best]).sum()),
        })
    table = pd.DataFrame(rows)

    out_dir = split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis" / "archetype" / "unit_combinations",
                            args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.family}_L{args.L}_unit_combinations.csv"
    table.to_csv(csv_path, index=False)
    print(f"{args.family} L={args.L} ({args.split}), seed {seed_dir.name}, "
          f"{len(days)} days, {len(freq)} patterns used")
    print(table.round(3).to_string(index=False))
    print(f"Saved: {csv_path}")

    if args.tex_out:
        args.tex_out.write_text(to_latex(table))
        print(f"Saved: {args.tex_out}")


def to_latex(table: pd.DataFrame) -> str:
    """Rows = quantities, columns = archetypes; row meanings go in the caption."""
    def row(label, values):
        return f"    {label} & " + " & ".join(values) + r" \\" + "\n"

    cols = "l" + "c" * len(table)
    body = (
        row("No unit", [f"{v:.2f}" for v in table["baseline_cos"]])
        + row("Best unit", [f"{v:.2f}" for v in table["single_cos"]])
        + row("Best combin.", [f"{v:.2f}" for v in table["combination_cos"]])
        + row(r"\quad pattern",
              [rf"\texttt{{{p}}}" for p in table["combination"]])
        + row(r"\quad rank",
              [f"{int(r)}" for r in table["frequency_rank"]])
        + row(r"\quad days in it",
              [f"{v:.0%}".replace("%", r"\%") for v in table["share_of_archetype_days"]])
    )
    return (
        "% generated by code/scripts/analysis/archetype/archetype_unit_combinations.py\n"
        r"\begin{tabular}{" + cols + "}\n"
        r"    \toprule" + "\n"
        + row("", list(table["archetype"]))
        + r"    \midrule" + "\n"
        + body
        + r"    \bottomrule" + "\n"
        r"\end{tabular}" + "\n"
    )


if __name__ == "__main__":
    main()
