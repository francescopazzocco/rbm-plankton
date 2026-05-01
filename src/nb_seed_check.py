"""Per-seed final val metric for all families across L values."""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results" / "multiseed_pcd"

# metric column and display label per family prefix
METRIC = {
    "nb":               ("val_nll", "val_nll"),
    "bernoulli_median": ("val_pll", "val_pll"),
    "bernoulli_zero":   ("val_pll", "val_pll"),
}


def check_family(family_dirs, col, label, n_seeds):
    print(f"\n{'='*55}")
    print(f"  {label}  ({col}, lower is better)")
    print(f"{'='*55}")
    for d in family_dirs:
        l_val = d.name.split("_L")[1]
        vals = []
        print(f"\n  L={l_val}")
        print(f"  {'seed':<10} {col:>10}  status")
        print(f"  {'-'*34}")
        for seed_dir in sorted(d.glob("seed_*")):
            seed = seed_dir.name
            csv = seed_dir / "rbm_training_curves.csv"
            if not csv.exists():
                print(f"  {seed:<10} {'—':>10}  MISSING")
                continue
            df = pd.read_csv(csv)
            series = df[col].dropna() if col in df.columns else pd.Series([], dtype=float)
            if series.empty:
                print(f"  {seed:<10} {'—':>10}  DIV")
            else:
                v = series.iloc[-1]
                vals.append(v)
                print(f"  {seed:<10} {v:>10.4f}")
        if vals:
            s = pd.Series(vals)
            print(f"  {'summary':<10} {s.mean():>10.4f}  "
                  f"min={s.min():.4f}  max={s.max():.4f}  "
                  f"std={s.std():.4f}  n={len(vals)}/{n_seeds}")


def main():
    for family_prefix, (col, label) in METRIC.items():
        dirs = sorted(RESULTS_DIR.glob(f"{family_prefix}_L*"))
        if dirs:
            check_family(dirs, col, label, n_seeds=10)
        else:
            print(f"\n[{family_prefix}] no results found in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
