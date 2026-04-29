"""
main.py — Entry point for RBM plankton training pipeline
======================================================
"""

import os
from config import (
    DATA_PATH, OUTPUT_DIR, N_HIDDEN, EPOCHS, LR, LR_DECAY,
    CD_STEPS, N_BATCHES, BATCH_I, BATCH_F, GAMMA, BETA, EPSILON,
    VAL_FRAC, PLOT_RESULTS, VISIBLE_MODEL, BINARIZE_THRESHOLD,
    COUNT_SCALE, THETA_INIT_LOG
)
from data import load_and_binarise, load_raw_counts
from models import BernoulliRBM, NBRBM
from visualization import plot_training_curves, plot_weight_heatmap, plot_hidden_activations, export_results_csv
from utils import get_device


if __name__ == "__main__":

    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[Config]  VISIBLE_MODEL='{VISIBLE_MODEL}'  L={N_HIDDEN}  "
          f"epochs={EPOCHS}  lr={LR}  cd={CD_STEPS}")

    # ── Load data ────────────────────────────────────────────────
    if VISIBLE_MODEL == "bernoulli":
        X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds = \
            load_and_binarise(DATA_PATH, binarize=BINARIZE_THRESHOLD,
                              val_frac=VAL_FRAC, device=device)
        rbm = BernoulliRBM(n_visible=len(taxa_cols), n_hidden=N_HIDDEN, device=device)

    elif VISIBLE_MODEL == "nb":
        X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows = \
            load_raw_counts(DATA_PATH, scale=COUNT_SCALE,
                            val_frac=VAL_FRAC, device=device)
        rbm = NBRBM(n_visible=len(taxa_cols), n_hidden=N_HIDDEN,
                    device=device, theta_init_log=THETA_INIT_LOG)
        thresholds = None

    else:
        raise ValueError(f"Unknown VISIBLE_MODEL='{VISIBLE_MODEL}'. "
                         f"Use 'bernoulli' or 'nb'.")

    # ── Train ────────────────────────────────────────────────────
    history = rbm.train(
        X_train, X_val,
        epochs    = EPOCHS,
        lr        = LR,
        lr_decay  = LR_DECAY,
        cd_steps  = CD_STEPS,
        batch_i   = BATCH_I,
        batch_f   = BATCH_F,
        n_batches = N_BATCHES,
        gamma     = GAMMA,
        beta      = BETA,
        epsilon   = EPSILON,
        eval_every= 10,
        verbose   = True,
    )

    # ── Save ─────────────────────────────────────────────────────
    params = rbm.numpy_params()
    W, a, b = params[0], params[1], params[2]

    save_dict = dict(W=W, a=a, b=b, taxa=taxa_cols,
                     visible_model=VISIBLE_MODEL)
    if VISIBLE_MODEL == "nb":
        save_dict["log_theta"] = params[3]
    if thresholds is not None:
        save_dict["thresholds"] = thresholds

    from utils import save_weights
    save_weights(OUTPUT_DIR, save_dict)

    # ── Plots and CSV export ─────────────────────────────────────
    if PLOT_RESULTS:
        plot_training_curves(history, OUTPUT_DIR)
        plot_weight_heatmap(W, taxa_cols, OUTPUT_DIR)
        plot_hidden_activations(rbm, X_train, X_val,
                                dates_train, dates_val, OUTPUT_DIR)
    export_results_csv(history, W, taxa_cols, OUTPUT_DIR)

    # ── Final metrics ────────────────────────────────────────────
    print("\n[Final metrics]")
    print(f"  train MSE : {rbm.reconstruction_mse(X_train):.4f}")
    print(f"  val   MSE : {rbm.reconstruction_mse(X_val):.4f}")
    if VISIBLE_MODEL == "bernoulli":
        print(f"  train PLL : {rbm.pll(X_train):.4f}")
        print(f"  val   PLL : {rbm.pll(X_val):.4f}")
    if VISIBLE_MODEL == "nb":
        print(f"  train NLL : {rbm.nll(X_train):.4f}")
        print(f"  val   NLL : {rbm.nll(X_val):.4f}")
        theta_vals = rbm.log_theta.detach().exp().cpu().numpy()
        print(f"  θ range   : [{theta_vals.min():.3f}, {theta_vals.max():.3f}]  "
              f"mean={theta_vals.mean():.3f}")
        with torch.no_grad():
            ph = rbm.hidden_probs(X_train)
        sat_lo  = (ph < 0.1).float().mean().item()
        sat_hi  = (ph > 0.9).float().mean().item()
        sat_mid = 1.0 - sat_lo - sat_hi
        print(f"  h saturation: <0.1={sat_lo:.0%}  >0.9={sat_hi:.0%}  "
              f"mid={sat_mid:.0%}  →  "
              f"{'binary ✓' if sat_mid < 0.15 else 'not binary yet'}")
    print("\nDone.")
