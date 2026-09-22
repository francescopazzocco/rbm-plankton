"""
zinb_meanfield_test.py - Ancestral vs mean-field imputation for ZINB families.

Diagnostic for the open investigation in ROADMAP.md ("Now" #3): ZINB
consistently underperforms NB on NaN-imputation NLL (nan_eval_summary.csv).
Working hypothesis: score_row_gibbs's imputation loop redraws the discrete
z ~ Bernoulli(pi) zero/non-zero branch every Gibbs step (ancestral sampling
via _sample_zinb), which can anchor the chain to the zero-inflated mode. NB
has no such discrete branch and is not exposed to the same failure mode.

This script re-scores the same ZINB runs and rows as nan_test_eval.py, once
with the existing ancestral sample_zinb step and once with a deterministic
mean-field replacement (1-pi)*mu (meanfield_zinb in _eval_utils.py) inside the
imputation loop only -- the final scoring likelihood (loss_zinb, full ZINB
log-prob) is unchanged in both variants, so the comparison isolates the effect
of the imputation-loop sampling method, not the metric.

If mean-field imputation narrows or closes the NB-ZINB gap: supports the
mixing-artifact hypothesis. If the gap is unchanged: the ancestral zero/
non-zero coin flip is not the driver, and the deficit more likely reflects pi
mis-calibration or a genuine modelling limitation.

Not a canonical evaluation method -- exploratory only, output is untracked.

Output: diagnostic_outputs/zinb_meanfield_test/zinb_meanfield_rows.csv,
        diagnostic_outputs/zinb_meanfield_test/zinb_meanfield_summary.csv
"""

import pandas as pd

from models._eval_utils import loss_zinb, meanfield_zinb, sample_zinb, score_row_gibbs
from models.io import (
    COUNT_SCALE, METRIC_COL, SHUFFLED, best_seed_dir, load_model, load_nan_rows,
    run_dir, scale_counts,
)
from models.paths import DIAGNOSTIC_ROOT, RUNS_ROOT
from models.utils import get_device

OUT_DIR = DIAGNOSTIC_ROOT / "zinb_meanfield_test"

N_SAMPLES      = 100
IMPUTE_BASE    = 5
IMPUTE_PER_NAN = 3

# Same (family, L, split) specs nan_test_eval.py uses for the ZINB branch, so
# the comparison lines up with the existing nan_eval_summary.csv rows.
_SPECS = [
    ("zinb",         8, SHUFFLED),
    ("zinb_sigmoid", 7, SHUFFLED),
    ("zinb_softmax", 6, SHUFFLED),
]

PATTERN_MAP = {3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"}


def score_variant(rbm, v_raw, device, sample_visible):
    return score_row_gibbs(
        rbm, v_raw, device,
        n_samples=N_SAMPLES,
        impute_base=IMPUTE_BASE,
        impute_per_nan=IMPUTE_PER_NAN,
        sample_hidden=rbm._sample_bernoulli,
        sample_visible=sample_visible,
        compute_loss=loss_zinb)


def evaluate(rbm, rows_df, taxa, device, sample_visible):
    records = []
    for _, row in rows_df.iterrows():
        v = row[taxa].values.astype("float32")
        n_miss = int(pd.isna(v).sum())
        records.append({
            "date":   row["date"],
            "n_obs":  len(v) - n_miss,
            "n_miss": n_miss,
            "nll":    score_variant(rbm, v, device, sample_visible),
        })
    return pd.DataFrame(records)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    nan_df, taxa = load_nan_rows()
    count_rows = scale_counts(nan_df, taxa, COUNT_SCALE)

    all_rows = []
    for family, n_hidden, split in _SPECS:
        fam_dir = run_dir(family, n_hidden, split, RUNS_ROOT)
        if not fam_dir.exists():
            print(f"[SKIP] {fam_dir.name} not found")
            continue
        seed_dir = best_seed_dir(fam_dir, METRIC_COL[family])
        if seed_dir is None:
            print(f"[SKIP] no valid seed in {fam_dir.name}")
            continue

        rbm, _ = load_model(seed_dir, device)
        run_id = f"{family}_L{n_hidden}_{split}"
        print(f"\n=== {run_id} (seed: {seed_dir.name}) ===")

        for variant, sample_visible in [("ancestral", sample_zinb),
                                         ("meanfield", meanfield_zinb)]:
            df = evaluate(rbm, count_rows, taxa, device, sample_visible)
            df["run"] = run_id
            df["variant"] = variant
            all_rows.append(df)
            print(f"  {variant:10s}: mean NLL = {df['nll'].mean():.4f}")

    if not all_rows:
        print("\nNo runs evaluated -- nothing found under", RUNS_ROOT)
        return

    df = pd.concat(all_rows, ignore_index=True)
    df["pattern"] = df["n_miss"].map(PATTERN_MAP)

    rows_out = OUT_DIR / "zinb_meanfield_rows.csv"
    df.to_csv(rows_out, index=False)

    summary = (df.groupby(["run", "variant", "pattern", "n_miss", "n_obs"])
               .agg(n_rows=("nll", "count"), nll_mean=("nll", "mean"),
                    nll_std=("nll", "std"))
               .reset_index().sort_values(["run", "pattern", "variant"]))
    summary_out = OUT_DIR / "zinb_meanfield_summary.csv"
    summary.to_csv(summary_out, index=False)

    pivot = summary.pivot_table(index=["run", "pattern"], columns="variant",
                                 values="nll_mean").reset_index()
    if "ancestral" in pivot and "meanfield" in pivot:
        pivot["delta (meanfield - ancestral)"] = pivot["meanfield"] - pivot["ancestral"]

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- Ancestral vs mean-field ZINB imputation ---")
    print(pivot.to_string(index=False))
    print(f"\nRows saved:    {rows_out}")
    print(f"Summary saved: {summary_out}")


if __name__ == "__main__":
    main()
