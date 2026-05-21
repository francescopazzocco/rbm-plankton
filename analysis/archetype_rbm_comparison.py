#!/usr/bin/env python3
"""Quantitative evidence for the archetype vs. RBM comparison report.

Reads Cheng's archetype profiles (prof/) and the precomputed RBM
visible-by-hidden CSVs, seasonal profiles, state frequencies, and mean
activation summaries.  Prints all tables used in
doc/archetype_rbm_comparison.md.

Usage:
    python analysis/archetype_rbm_comparison.py [--top N]

Output goes to stdout; redirect to a file if you want to keep it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROF_PROFILES   = ROOT / "prof" / "archetypes_k5_profiles.csv"
PROF_TIMESERIES = ROOT / "prof" / "archetypes_k5_timeseries.csv"

VBH = {
    "nb_chrono":   ROOT / "analysis/results/nb_chrono_vbh/visible_by_hidden_bernoulli.csv",
    "zinb_chrono": ROOT / "analysis/results/zinb_chrono_vbh/visible_by_hidden_zinb.csv",
    "nb_shuffle":  ROOT / "analysis/results/nb_shuffle_vbh/visible_by_hidden_bernoulli.csv",
    "zinb_shuffle":ROOT / "analysis/results/zinb_shuffle_vbh/visible_by_hidden_zinb.csv",
}

SEASONAL_NB = ROOT / "results/tables/hidden/seasonal_profiles_nb.csv"
SEASONAL_BB = ROOT / "results/tables/hidden/seasonal_profiles_bb.csv"
STATE_FREQ  = ROOT / "results/02_model_analysis/state_frequency.csv"
MEAN_ACT    = ROOT / "results/02_model_analysis/mean_activation_summary.csv"


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ---------------------------------------------------------------------------
# 1. Cheng's archetype dominant taxa
# ---------------------------------------------------------------------------
def archetype_dominant_taxa(top_n: int = 5) -> None:
    separator("1. Cheng's archetypes — dominant taxa")
    df = pd.read_csv(PROF_PROFILES, index_col=0)
    for arch in df.index:
        row = df.loc[arch].sort_values(ascending=False).head(top_n)
        print(f"\n{arch}:")
        for taxon, val in row.items():
            print(f"  {taxon:<40s} {val:.4f}")


# ---------------------------------------------------------------------------
# 2. Archetype weight fraction captured by single dominant taxon
# ---------------------------------------------------------------------------
def archetype_dominance_ratio() -> None:
    separator("2. Fraction of weight in top taxon per archetype (dominance check)")
    df = pd.read_csv(PROF_PROFILES, index_col=0)
    for arch in df.index:
        row = df.loc[arch]
        top_val  = row.max()
        top_tax  = row.idxmax()
        total    = row.sum()
        print(f"  {arch}: {top_tax} = {top_val:.4f} / total {total:.4f} "
              f"({100*top_val/total:.1f}%)")


# ---------------------------------------------------------------------------
# 3. Archetype timeseries — fraction of days each archetype is dominant
# ---------------------------------------------------------------------------
def archetype_dominance_days() -> None:
    separator("3. Cheng's archetypes — fraction of days as dominant state")
    ts = pd.read_csv(PROF_TIMESERIES, index_col=0, parse_dates=True)
    dominant = ts.idxmax(axis=1)
    counts   = dominant.value_counts().sort_index()
    total    = len(dominant)
    for arch, n in counts.items():
        print(f"  {arch}: {n} days ({100*n/total:.1f}%)")


# ---------------------------------------------------------------------------
# 4. RBM visible-by-hidden — top taxa per hidden unit
# ---------------------------------------------------------------------------
def rbm_top_taxa(top_n: int = 5) -> None:
    separator(f"4. RBM visible-by-hidden — top {top_n} taxa per hidden unit")
    for label, path in VBH.items():
        print(f"\n--- {label} ({path.name}) ---")
        df = pd.read_csv(path, index_col=0)
        for col in df.columns:
            ranked = df[col].sort_values(ascending=False).head(top_n)
            entries = ", ".join(f"{t}={v:.3f}" for t, v in ranked.items())
            print(f"  {col}: {entries}")


# ---------------------------------------------------------------------------
# 5. Cryptophyte distribution across hidden units (NB vs ZINB)
# ---------------------------------------------------------------------------
def cryptophyte_distribution() -> None:
    separator("5. Cryptophyte value across hidden units — NB vs. ZINB chrono")
    for label in ("nb_chrono", "zinb_chrono"):
        df = pd.read_csv(VBH[label], index_col=0)
        if "cryptophyte" not in df.index:
            print(f"  [{label}] cryptophyte not found in index")
            continue
        row = df.loc["cryptophyte"]
        print(f"\n  {label}  (scale: {'probability' if 'nb_chrono' == label else 'expected count'})")
        for unit, val in row.items():
            print(f"    {unit}: {val:.4f}")
        print(f"    range: [{row.min():.4f}, {row.max():.4f}]  "
              f"std: {row.std():.4f}")


# ---------------------------------------------------------------------------
# 6. Key taxa across NB chrono units (aulacoseira, dinobryon, uroglena,
#    centric_diatom, chlorophyte)
# ---------------------------------------------------------------------------
def key_taxa_profiles() -> None:
    separator("6. Key taxon values across NB chrono hidden units")
    df = pd.read_csv(VBH["nb_chrono"], index_col=0)
    focal = ["aulacoseira", "centric_diatom", "dinobryon", "uroglena",
             "chlorophyte", "chlorophyte_colonial_dividing",
             "cyanobacteria_colonial_probably", "oocystaceae", "cryptophyte",
             "rhodomonas"]
    for taxon in focal:
        if taxon not in df.index:
            continue
        row = df.loc[taxon]
        vals = "  ".join(f"{u}={v:.3f}" for u, v in row.items())
        print(f"  {taxon:<38s} {vals}")


# ---------------------------------------------------------------------------
# 7. Seasonal profiles — NB and BB (monthly mean activation per unit)
# ---------------------------------------------------------------------------
def seasonal_profiles() -> None:
    separator("7. Seasonal profiles — mean monthly activation per hidden unit")
    for label, path in (("NB chrono L=6", SEASONAL_NB),
                        ("Bernoulli-median chrono L=6", SEASONAL_BB)):
        df = pd.read_csv(path, index_col=0)
        print(f"\n  {label}")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        df.index = month_names
        print(df.round(3).to_string())


# ---------------------------------------------------------------------------
# 8. State frequency — dominant-state utilisation per model/L
# ---------------------------------------------------------------------------
def state_frequency(families: list[str] | None = None,
                    L_values: list[int] | None = None) -> None:
    separator("8. State frequency — fraction of days each unit is dominant")
    df = pd.read_csv(STATE_FREQ)
    if families:
        df = df[df["family"].isin(families)]
    if L_values:
        df = df[df["L"].isin(L_values)]
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 9. Mean activation summary — flag absorber units
# ---------------------------------------------------------------------------
def absorber_summary(families: list[str] | None = None,
                     L_values: list[int] | None = None) -> None:
    separator("9. Mean activation summary — absorber flags")
    df = pd.read_csv(MEAN_ACT)
    if families:
        df = df[df["family"].isin(families)]
    if L_values:
        df = df[df["L"].isin(L_values)]
    # highlight absorbers
    absorbers = df[df["flag"] != "active"]
    actives   = df[df["flag"] == "active"]
    print(f"  Total units: {len(df)}")
    print(f"  Active: {len(actives)}  |  Absorbers: {len(absorbers)}")
    if len(absorbers):
        print("\n  Absorber units:")
        print(absorbers.to_string(index=False))

    # per-family absorber rate
    print("\n  Absorber rate by family:")
    for fam, grp in df.groupby("family"):
        n_abs = (grp["flag"] != "active").sum()
        print(f"    {fam:<25s} {n_abs}/{len(grp)} absorbers "
              f"({100*n_abs/len(grp):.0f}%)")


# ---------------------------------------------------------------------------
# 10. NB L=6 degeneration check
# ---------------------------------------------------------------------------
def nb_l6_degeneration() -> None:
    separator("10. NB L=6 chrono degeneration — dominant-state breakdown")
    df = pd.read_csv(STATE_FREQ)
    sub = df[(df["family"] == "nb") & (df["L"] == 6)]
    print(sub.to_string(index=False))
    ma = pd.read_csv(MEAN_ACT)
    sub_ma = ma[(ma["family"] == "nb") & (ma["L"] == 6)]
    print()
    print(sub_ma.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top taxa to show per unit (default: 5)")
    parser.add_argument("--families", nargs="*",
                        help="Restrict state-frequency/absorber tables to these families")
    parser.add_argument("--L", nargs="*", type=int,
                        help="Restrict state-frequency/absorber tables to these L values")
    args = parser.parse_args()

    archetype_dominant_taxa(top_n=args.top)
    archetype_dominance_ratio()
    archetype_dominance_days()
    rbm_top_taxa(top_n=args.top)
    cryptophyte_distribution()
    key_taxa_profiles()
    seasonal_profiles()
    state_frequency(families=args.families, L_values=args.L)
    absorber_summary(families=args.families, L_values=args.L)
    nb_l6_degeneration()


if __name__ == "__main__":
    main()
