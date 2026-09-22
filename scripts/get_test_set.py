"""
get_test_set.py

Extract the test set (validation-sized) from the project's CSV.
Behavior:
 - Non-shuffled: return the last `n_val` rows (chronological split).
 - Shuffled: return a random set of `n_val` rows (reproducible with --seed).

Usage examples:
  python scripts/get_test_set.py --out test_rows.csv         # chronological (default)
  python scripts/get_test_set.py --shuffle --seed 0 --out test_rows.csv
  python scripts/get_test_set.py --shuffle --seed 0 --indices  # print indices only

"""

import argparse
from pathlib import Path
import numpy as np

# Import internal helper from project
from src.models.io import _base_load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=str, default='data/raw/TimeSeries_countsuL_clean.csv')
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--shuffle', action='store_true', help='Return a random test set of size n_val')
    p.add_argument('--seed', type=int, default=None, help='Random seed for shuffled selection')
    p.add_argument('--out', type=str, default=None, help='Output CSV path for test rows')
    p.add_argument('--indices', action='store_true', help='Print indices of selected rows instead of rows')
    args = p.parse_args()

    path = args.data_path
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Use _base_load to parse, sort and drop NaN / zero rows (same preprocessing as training)
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = _base_load(path, args.val_frac, device=None, shuffle=False)

    n_total = len(df)
    n_val = int(n_total * args.val_frac)

    if not args.shuffle:
        # Chronological: last n_val rows
        test_df = df.iloc[n_train:]
        if args.indices:
            print(list(test_df.index))
        else:
            if args.out:
                test_df.to_csv(args.out, index=False)
                print(f"Wrote {len(test_df)} rows → {args.out}")
            else:
                print(test_df.head())
                print(f"Selected {len(test_df)} test rows (chronological).")
    else:
        # Random selection of n_val rows
        if args.seed is not None:
            np.random.seed(args.seed)
        else:
            # seed for unpredictable behavior
            np.random.seed(None)

        chosen = np.random.choice(df.index.to_numpy(), size=n_val, replace=False)
        test_df = df.loc[chosen]

        if args.indices:
            print(list(chosen))
        else:
            if args.out:
                test_df.to_csv(args.out, index=False)
                print(f"Wrote {len(test_df)} random test rows → {args.out} (seed={args.seed})")
            else:
                print(test_df.head())
                print(f"Selected {len(test_df)} random test rows (seed={args.seed}).")


if __name__ == '__main__':
    main()
