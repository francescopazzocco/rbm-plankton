"""
main_multiseed.py - Parallel multi-seed training for statistical L comparison.

Trains each (family, L, seed) combination in parallel using ProcessPoolExecutor.
Saves CSVs and weights to results/training_runs/{family}_L{n}/seed_{k}/.

Usage:
    python src/main_multiseed.py
"""

import contextlib
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# -- Sweep configuration -------------------------------------------------------

N_SEEDS     = 10
MAX_WORKERS = 10

L_VALUES = {
    "nb":               [3, 4, 5, 6, 7],
    "bernoulli_median": [3, 4, 5, 6, 7],
    "bernoulli_zero":   [3, 4, 5, 6, 7],
}

# Fixed hyperparameters
EPOCHS         = 500
LR             = 0.01
LR_DECAY       = 0.998
CD_STEPS       = 1
N_BATCHES      = 20
BATCH_I        = 10
BATCH_F        = 256
GAMMA          = 1e-4
BETA           = 0.9
EPSILON        = 1e-4
VAL_FRAC       = 0.15
COUNT_SCALE    = 1000
THETA_INIT_LOG = 0.0

# PCD settings (NB only - Bernoulli landscape is well-conditioned)
USE_PCD      = True
N_PCD_CHAINS = 500   # must be >= BATCH_F

DATA_PATH = Path(__file__).parent.parent / "data/raw/TimeSeries_countsuL_clean.csv"
OUT_ROOT  = Path(__file__).parent.parent / "results" / "training_runs"


# -- Worker --------------------------------------------------------------------

def train_one(job: tuple) -> str:
    family, l_val, seed, out_dir = job

    # Redirect stdout to a per-run log so parallel workers don't interleave
    log_path = Path(out_dir) / "train.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log, contextlib.redirect_stdout(log):
        try:
            import torch
            import numpy as np
            import pandas as pd

            torch.manual_seed(seed)
            np.random.seed(seed)

            from models.io import load_and_binarise, load_raw_counts
            from models import BernoulliRBM, NBRBM
            from models.visualization import export_results_csv
            from models.utils import get_device, save_weights

            device = get_device()

            if family.startswith("bernoulli"):
                threshold = "median" if "median" in family else "zero"
                X_train, X_val, dates_train, dates_val, taxa_cols, _, thresholds = \
                    load_and_binarise(str(DATA_PATH), binarize=threshold,
                                      val_frac=VAL_FRAC, device=device)
                rbm = BernoulliRBM(n_visible=len(taxa_cols), n_hidden=l_val,
                                   device=device)
            else:
                X_train, X_val, dates_train, dates_val, taxa_cols, _ = \
                    load_raw_counts(str(DATA_PATH), scale=COUNT_SCALE,
                                    val_frac=VAL_FRAC, device=device)
                rbm = NBRBM(n_visible=len(taxa_cols), n_hidden=l_val,
                            device=device, theta_init_log=THETA_INIT_LOG)
                thresholds = None

            pcd_kwargs = ({"use_pcd": USE_PCD, "n_pcd_chains": N_PCD_CHAINS}
                          if family == "nb" else {})
            history = rbm.train(
                X_train, X_val,
                epochs=EPOCHS, lr=LR, lr_decay=LR_DECAY,
                cd_steps=CD_STEPS, batch_i=BATCH_I, batch_f=BATCH_F,
                n_batches=N_BATCHES, gamma=GAMMA, beta=BETA, epsilon=EPSILON,
                eval_every=10, verbose=False,
                **pcd_kwargs,
            )

            # Save training curves + weights
            params = rbm.numpy_params()
            W, a, b = params[0], params[1], params[2]
            save_dict = dict(W=W, a=a, b=b, taxa=taxa_cols,
                             visible_model=family)
            if family == "nb":
                save_dict["log_theta"] = params[3]
            if thresholds is not None:
                save_dict["thresholds"] = thresholds
            save_weights(out_dir, save_dict)
            export_results_csv(history, W, taxa_cols, out_dir)

            # Save hidden activations CSV (no plot)
            with torch.no_grad():
                H = torch.cat([rbm.hidden_probs(X_train),
                               rbm.hidden_probs(X_val)], dim=0).cpu().numpy()
            dates_all = pd.concat([dates_train, dates_val]).reset_index(drop=True)
            df_h = pd.DataFrame(H, columns=[f"h{j}" for j in range(l_val)])
            df_h.insert(0, "date", dates_all.values)
            df_h.to_csv(Path(out_dir) / "rbm_hidden_activations.csv", index=False)

        except Exception as e:
            return f"ERR  {family} L={l_val} seed={seed}: {e}"

    return f"OK   {family} L={l_val} seed={seed}"


# -- Main ----------------------------------------------------------------------

def build_jobs() -> list[tuple]:
    jobs = []
    for family, l_list in L_VALUES.items():
        for l_val in l_list:
            for seed in range(N_SEEDS):
                out_dir = OUT_ROOT / f"{family}_L{l_val}" / f"seed_{seed}"
                # Skip if already completed
                if (out_dir / "rbm_training_curves.csv").exists():
                    continue
                jobs.append((family, l_val, seed, str(out_dir)))
    return jobs


def main():
    jobs = build_jobs()
    total = sum(
        len(l_list) * N_SEEDS
        for l_list in L_VALUES.values()
    )
    print(f"Total runs planned : {total}")
    print(f"Already completed  : {total - len(jobs)}")
    print(f"To run             : {len(jobs)}  (max_workers={MAX_WORKERS})")

    if not jobs:
        print("Nothing to do.")
        return

    completed = 0
    failed    = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(train_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("OK"):
                completed += 1
            else:
                failed += 1
            print(f"[{completed + failed:>3}/{len(jobs)}]  {result}")

    print(f"\nDone. {completed} OK  |  {failed} failed")


if __name__ == "__main__":
    main()
