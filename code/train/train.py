"""
train.py - Train RBM-plankton models.
======================================
Default mode: single run (SINGLE_RUN=True in config.py).
  Trains one model and prints progress to stdout.
  No parallelism, no multi-seed — just make me a model.

Sweep mode: SINGLE_RUN=False in config.py.
  Trains all (family, L, seed) combinations in parallel.

Usage:
    python code/train/train.py
    python code/train/train.py --family nb_sigmoid --L 6   # CLI override
"""

import argparse
import contextlib
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    BATCH_F, BATCH_I, BETA, COUNT_SCALE, CD_STEPS, DATA_PATH,
    EPSILON, EPOCHS, EVAL_EVERY, GAMMA, LR, LR_DECAY, LR_PARAM_MULTIPLIER,
    L_VALUES, MAX_WORKERS, N_BATCHES, N_PCD_CHAINS, N_SEEDS,
    OUT_ROOT, SHUFFLE_SPLIT, SHUFFLE_TAG, SINGLE_RUN,
    SINGLE_RUN_FAMILY, SINGLE_RUN_L, SINGLE_RUN_SEED,
    THETA_INIT_LOG, USE_PCD, VAL_FRAC,
)


PCD_FAMILIES = {"nb", "zinb", "nb_relu", "zinb_relu",
                "nb_sigmoid", "nb_softmax",
                "zinb_sigmoid", "zinb_softmax"}


# -- Model constructors --------------------------------------------------------

_MODEL_REGISTRY: dict[str, tuple] = {
    "bernoulli_median": ("bernoulli", "load_and_binarise", "BernoulliRBM", {"binarize": "median"}),
    "bernoulli_zero":   ("bernoulli", "load_and_binarise", "BernoulliRBM", {"binarize": "zero"}),
    "nb":               ("counts",    "load_raw_counts",   "NB_RBM",       {}),
    "zinb":             ("counts",    "load_raw_counts",   "ZINB_RBM",     {}),
    "nb_relu":          ("counts",    "load_raw_counts",   "NB_ReLU_RBM",  {}),
    "zinb_relu":        ("counts",    "load_raw_counts",   "ZINB_ReLU_RBM",{}),
    "nb_sigmoid":       ("counts",    "load_raw_counts",   "NBSigmoidRBM",  {}),
    "nb_softmax":       ("counts",    "load_raw_counts",   "NBSoftmaxRBM",  {}),
    "zinb_sigmoid":     ("counts",    "load_raw_counts",   "ZINBSigmoidRBM",{}),
    "zinb_softmax":     ("counts",    "load_raw_counts",   "ZINBSoftmaxRBM",{}),
}


# -- Worker --------------------------------------------------------------------

def train_one(job: tuple) -> str:
    family, l_val, seed, out_dir = job

    log_path = Path(out_dir) / "train.log"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log, contextlib.redirect_stdout(log):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            import torch
            import numpy as np
            import pandas as pd

            torch.manual_seed(seed)
            np.random.seed(seed)

            from models.io import load_and_binarise, load_raw_counts
            from models import (BernoulliRBM, NB_RBM, NB_ReLU_RBM,
                                NBSigmoidRBM, NBSoftmaxRBM,
                                ZINB_RBM, ZINB_ReLU_RBM,
                                ZINBSigmoidRBM, ZINBSoftmaxRBM)
            from models.visualization import export_results_csv
            from models.utils import get_device, save_weights

            device = get_device()
            loader_cfg = _MODEL_REGISTRY[family]
            data_mode = loader_cfg[0]
            loader_fn = loader_cfg[1]
            model_cls_name = loader_cfg[2]
            loader_kw = dict(loader_cfg[3])
            shuffle_kw = dict(shuffle=SHUFFLE_SPLIT)

            model_cls = locals()[model_cls_name]
            is_pcd = family in PCD_FAMILIES

            if data_mode == "bernoulli":
                X_train, X_val, dates_train, dates_val, taxa_cols, _, thresholds = \
                    load_and_binarise(str(DATA_PATH), scale=COUNT_SCALE,
                                      val_frac=VAL_FRAC, device=device,
                                      **loader_kw, **shuffle_kw)
                rbm = model_cls(n_visible=len(taxa_cols), n_hidden=l_val, device=device)
            else:
                X_train, X_val, dates_train, dates_val, taxa_cols, _ = \
                    load_raw_counts(str(DATA_PATH), scale=COUNT_SCALE,
                                    val_frac=VAL_FRAC, device=device,
                                    **shuffle_kw)
                rbm = model_cls(n_visible=len(taxa_cols), n_hidden=l_val,
                                device=device, theta_init_log=THETA_INIT_LOG)
                thresholds = None

            pcd_kwargs = {"use_pcd": USE_PCD, "n_pcd_chains": N_PCD_CHAINS} if is_pcd else {}

            history = rbm.train(
                X_train, X_val,
                epochs=EPOCHS, lr=LR, lr_decay=LR_DECAY,
                cd_steps=CD_STEPS, batch_i=BATCH_I, batch_f=BATCH_F,
                n_batches=N_BATCHES, gamma=GAMMA, beta=BETA,
                epsilon=EPSILON, eval_every=EVAL_EVERY, verbose=False,
                **pcd_kwargs,
            )

            params = rbm.numpy_params()
            W, a, b = params[0], params[1], params[2]
            save_dict = dict(W=W, a=a, b=b, taxa=taxa_cols, visible_model=family)
            if family in ("nb", "nb_relu", "nb_sigmoid", "nb_softmax"):
                save_dict["log_theta"] = params[3]
            if family in ("zinb", "zinb_relu", "zinb_sigmoid", "zinb_softmax"):
                save_dict["log_theta"] = params[3]
                save_dict["logit_pi"] = params[4]
            if thresholds is not None:
                save_dict["thresholds"] = thresholds
            save_weights(out_dir, save_dict)
            export_results_csv(history, W, taxa_cols, out_dir)

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


# -- Job construction ----------------------------------------------------------

def build_jobs() -> list[tuple]:
    if SINGLE_RUN:
        families = {SINGLE_RUN_FAMILY: [SINGLE_RUN_L]}
        n_seeds = 1
    else:
        families = L_VALUES
        n_seeds = N_SEEDS

    shuffle_tag = SHUFFLE_TAG if SHUFFLE_SPLIT else ""
    jobs = []
    for family, l_list in families.items():
        for l_val in l_list:
            for seed in range(n_seeds):
                seed_id = seed if not SINGLE_RUN else SINGLE_RUN_SEED
                out_dir = OUT_ROOT / f"{family}_L{l_val}{shuffle_tag}" / f"seed_{seed_id}"
                if (out_dir / "rbm_training_curves.csv").exists():
                    continue
                jobs.append((family, l_val, seed_id, str(out_dir)))
    return jobs


# -- CLI -----------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RBM-plankton models. "
                    "Edits config.py to change hyperparameters; "
                    "--family/--L/--seeds override config at runtime.")
    parser.add_argument("--family", type=str, default=None,
                        help="Override family (overrides SINGLE_RUN_FAMILY)")
    parser.add_argument("--L", type=int, default=None,
                        help="Override n_hidden (overrides SINGLE_RUN_L)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of seeds (default: config N_SEEDS)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to globals for job construction
    global SINGLE_RUN, SINGLE_RUN_FAMILY, SINGLE_RUN_L, SINGLE_RUN_SEED, N_SEEDS

    if args.family:
        SINGLE_RUN = True
        SINGLE_RUN_FAMILY = args.family
    if args.L is not None:
        SINGLE_RUN = True
        SINGLE_RUN_L = args.L
    if args.seeds is not None:
        N_SEEDS = args.seeds

    jobs = build_jobs()

    if SINGLE_RUN:
        print(f"Single run: {SINGLE_RUN_FAMILY} L={SINGLE_RUN_L} seed={SINGLE_RUN_SEED}")
    else:
        total = sum(len(l_list) * N_SEEDS for l_list in L_VALUES.values())
        already = total - len(jobs)
        print(f"Total runs planned  : {total}")
        print(f"Already completed   : {already}")
    print(f"To run              : {len(jobs)}")

    if not jobs:
        print("Nothing to do.")
        return

    if SINGLE_RUN:
        result = train_one(jobs[0])
        print(result)
    else:
        completed = 0
        failed = 0
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
