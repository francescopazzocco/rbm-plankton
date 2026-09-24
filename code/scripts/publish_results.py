"""publish_results.py

Move staged output from diagnostic_outputs/ into the tracked results/ tree,
recording provenance in results/MANIFEST.json (models.manifest.publish).

results/ is never written to directly by any other script (see
ARCHITECTURE.md, DECISION_LOG LOG-025). This is the only door in.

Usage:
    python code/scripts/publish_results.py 01_exploratory
    python code/scripts/publish_results.py --all
    python code/scripts/publish_results.py --src reconstruction_plots --dst reconstruction_plots \\
        --by "code/scripts/analysis/reconstruction/use_trained_rbm.py" --pattern "*.png"
"""

from __future__ import annotations

import argparse

from models.manifest import publish
from models.paths import DIAGNOSTIC_ROOT

# name -> (staging subpath under diagnostic_outputs/, results subpath, glob
# pattern, producer label). One diagnostic_outputs/ source can fan out to
# several results/ destinations (nan_test_eval writes csv+png side by side,
# but only the csv half is report evidence, the png half another) — that is
# why pattern is per-entry rather than always "**/*".
CATEGORIES = {
    "01_exploratory": (
        "01_exploratory", "01_exploratory", "**/*",
        "code/scripts/train/dataset_analysis.py"),
    "02_model_analysis": (
        "02_model_analysis", "02_model_analysis", "**/*",
        "code/scripts/analysis/hidden/ (hidden_dominant_state.py, hidden_mean_activation.py, "
        "hidden_cross_model.py, hidden_pattern_analysis.py, rbm_hidden_stackplot.py, "
        "plot_visible_by_hidden.py), code/scripts/analysis/archetype/ (archetype_rbm_comparison.py, "
        "distance_archetypes_rbm.py, overlap_archetypes_rbm.py, archetype_closest_rbm_scatter.py)"),
    "tables": (
        "tables", "tables", "**/*",
        "code/scripts/diagnostic/split_comparison.py, code/scripts/analysis/hidden/hidden_cross_model.py"),
    "03_evaluation": (
        "03_evaluation", "03_evaluation", "**/*",
        "code/scripts/diagnostic/split_comparison.py"),
    "04_model_selection": (
        "04_model_selection", "04_model_selection", "**/*",
        "code/scripts/diagnostic/sweep_analysis.py, code/scripts/archive/plot_final_metric_nb.py"),
    "diagnostics_all_families": (
        "diagnostics/training_curves/all_families_by_L", "diagnostics/training_curves/all_families_by_L", "**/*",
        "code/scripts/diagnostic/sweep_analysis.py"),
    "diagnostics_nb_zinb_parameters": (
        "diagnostics/nb_zinb_parameters", "diagnostics/nb_zinb_parameters", "**/*",
        "code/scripts/diagnostic/sweep_analysis.py"),
    "diagnostics_single_family": (
        "diagnostics/training_curves/single_family_by_L", "diagnostics/training_curves/single_family_by_L", "**/*",
        "code/scripts/diagnostic/plot_train_nll_curves.py"),
    "diagnostics_family_comparison": (
        "diagnostics/training_curves/family_comparison_fixed_L", "diagnostics/training_curves/family_comparison_fixed_L", "**/*",
        "code/scripts/diagnostic/plot_nll_curves.py"),
    "nan_eval_tables": (
        "nan_eval_extended", "tables", "*.csv",
        "code/scripts/diagnostic/nan_test_eval.py"),
    "nan_eval_figures": (
        "nan_eval_extended", "03_evaluation", "*.png",
        "code/scripts/diagnostic/nan_test_eval.py"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", choices=sorted(CATEGORIES),
                         help="known category to publish")
    parser.add_argument("--all", action="store_true", help="publish every known category")
    parser.add_argument("--src", type=str, help="ad-hoc: staging path under diagnostic_outputs/")
    parser.add_argument("--dst", type=str, help="ad-hoc: destination subpath under results/")
    parser.add_argument("--by", type=str, help="ad-hoc: producer label to record")
    parser.add_argument("--pattern", type=str, default="**/*", help="ad-hoc glob (default: everything)")
    args = parser.parse_args()

    if args.all:
        jobs = list(CATEGORIES.items())
    elif args.name:
        jobs = [(args.name, CATEGORIES[args.name])]
    elif args.src and args.dst and args.by:
        jobs = [("(ad-hoc)", (args.src, args.dst, args.pattern, args.by))]
    else:
        parser.error("pass a category name, --all, or --src/--dst/--by")
        return

    for label, (src_sub, dst_sub, pattern, by) in jobs:
        src = DIAGNOSTIC_ROOT / src_sub
        if not src.exists():
            print(f"[SKIP] {label}: {src} does not exist")
            continue
        published = publish(src, dst_sub, by, pattern=pattern)
        print(f"[{label}] published {len(published)} file(s) to results/{dst_sub}/")


if __name__ == "__main__":
    main()
