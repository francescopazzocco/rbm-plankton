"""
rbm_plankton.py — Bernoulli-Bernoulli RBM on Lake Greifen plankton data
========================================================================

Adapted from rbm_train_export.py (originally MNIST/FPGA).

What changed from the original:
  - Visible units: binarised plankton counts (detected / not detected)
  - Data loader: CSV pipeline replacing MNIST/torchvision
  - Monitor: reconstruction MSE replaces Pseudo-Log-Likelihood (PLL is
    only valid for binary visible units trained on binary data — here
    we keep it since we binarise, but MSE is more interpretable)
  - Removed: FPGA export (.coe / .bin), MNIST loader, PLL monitor

What is unchanged:
  - CD-k training engine
  - RMSprop optimiser
  - Batch size annealing
  - Training loop structure

Architecture:
  Visible : 83  (plankton taxa, binarised)
  Hidden  : configurable via N_HIDDEN (start: 5)

See DECISIONS.md for all open branches.
"""

import os
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# =====================  CONFIGURABLE PARAMETERS  ===========================
# ---------------------------------------------------------------------------

DATA_PATH   = "data/raw/TimeSeries_countsuL_clean.csv"
OUTPUT_DIR  = "results"

N_HIDDEN    = 5             # start: 5 (archetype reference). sweep: 3, 7, 10
EPOCHS      = 500
LR          = 0.01
LR_DECAY    = 0.998
CD_STEPS    = 1             # CD-1 to start; try CD-2 if reconstruction is poor
N_BATCHES   = 20
BATCH_I     = 10            # initial batch size
BATCH_F     = 256           # final batch size
GAMMA       = 1e-4          # L1 regularisation (sparse hidden activations)
BETA        = 0.9           # RMSprop momentum
EPSILON     = 1e-4          # RMSprop numerical stability
VAL_FRAC    = 0.15          # last 15% of timeline → validation
HIDDEN_ACT  = "sigmoid"     # hidden activation: "sigmoid" or "relu"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BRANCH — Binarization threshold
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "zero"   : v > 0  (detected / not detected — presence/absence)
# "median" : v > per-taxon median  (above/below typical abundance)
#
# See DECISIONS.md for trade-off discussion.

BINARIZE_THRESHOLD = "median"     # ← SWITCH HERE: "zero" | "median"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PLOT_RESULTS = True

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_binarise(
    path: str,
    binarize: str = BINARIZE_THRESHOLD,
    val_frac: float = VAL_FRAC,
    device: torch.device = torch.device("cpu"),
):
    """
    Full preprocessing pipeline → binary torch tensors.

    Steps:
      1. Drop all-zero rows  (instrument downtime artifacts)
      2. Drop NaN rows       (ML classifier outage blocks → used as test set)
      3. Binarise            (BRANCH: "zero" or "median")
      4. Chronological split (no random shuffling — preserve temporal order)

    Returns
    -------
    X_train, X_val : torch.Tensor  float32, shape (N, 83), values in {0, 1}
    dates_train, dates_val : pd.Series
    taxa_cols : list[str]
    nan_rows  : pd.DataFrame  (structured test set for post-training evaluation)
    thresholds: np.ndarray    (per-taxon thresholds used for binarisation)
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa_cols = [c for c in df.columns if c != "date"]

    # Step 1 — drop all-zero rows
    nonzero_mask = df[taxa_cols].sum(axis=1) > 0
    df = df[nonzero_mask].copy()

    # Step 2 — separate NaN rows (save for post-training test)
    nan_mask  = df[taxa_cols].isna().any(axis=1)
    nan_rows  = df[nan_mask].copy()
    df        = df[~nan_mask].copy().reset_index(drop=True)

    print(f"[Data]  clean rows: {len(df)}  |  NaN rows (test set): {len(nan_rows)}")

    X = df[taxa_cols].values.astype(np.float32)

    # Step 3 — binarise
    if binarize == "zero":
        thresholds = np.zeros(len(taxa_cols), dtype=np.float32)
        X_bin = (X > 0).astype(np.float32)
        print(f"[Binarise]  threshold = 0  (presence/absence)")
    elif binarize == "median":
        thresholds = np.median(X, axis=0)
        X_bin = (X > thresholds).astype(np.float32)
        print(f"[Binarise]  threshold = per-taxon median")
    else:
        raise ValueError(f"Unknown BINARIZE_THRESHOLD='{binarize}'. Use 'zero' or 'median'.")

    # Warn about near-constant columns (little information for the RBM)
    col_means = X_bin.mean(axis=0)
    n_const = ((col_means < 0.02) | (col_means > 0.98)).sum()
    if n_const > 0:
        print(f"[Warning]  {n_const} taxa are >98% constant after binarisation "
              f"(contribute little information to visible layer)")

    # Step 4 — chronological split
    n_val   = int(len(X_bin) * val_frac)
    n_train = len(X_bin) - n_val

    X_train = X_bin[:n_train]
    X_val   = X_bin[n_train:]
    dates_train = df["date"].iloc[:n_train]
    dates_val   = df["date"].iloc[n_train:]

    print(f"[Split]  train: {n_train}  val: {n_val}  "
          f"(val starts {dates_val.iloc[0].date()})")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=device)

    return X_train_t, X_val_t, dates_train, dates_val, taxa_cols, nan_rows, thresholds


# ---------------------------------------------------------------------------
# RBM
# ---------------------------------------------------------------------------

class RBM:
    """
    Bernoulli-Bernoulli RBM trained by Contrastive Divergence (CD-k).

    Tensor shapes:
      W : (D, L)      weight matrix
      a : (D,)        visible bias
      b : (L,)        hidden bias
      V : (batch, D)  visible layer
      H : (batch, L)  hidden layer

    where D = n_visible = 83, L = n_hidden.
    """

    def __init__(self, n_visible: int, n_hidden: int,
                 device: torch.device = torch.device("cpu")):
        self.D      = n_visible
        self.L      = n_hidden
        self.device = device

        # Xavier-like initialisation
        scale  = math.sqrt(4.0 / (n_visible + n_hidden))
        self.W = torch.randn(n_visible, n_hidden, device=device) * scale
        self.b = torch.zeros(n_hidden,  device=device)
        self.a = torch.zeros(n_visible, device=device)

    # --- conditional distributions ---

    def _ph_given_v(self, V):
        """P(H=1 | V)  →  (batch, L)"""
        if HIDDEN_ACT == "relu":
            return F.relu(V @ self.W + self.b).clamp(0, 1)
        return torch.sigmoid(V @ self.W + self.b)

    def _pv_given_h(self, H):
        """P(V=1 | H)  →  (batch, D)"""
        return torch.sigmoid(H @ self.W.t() + self.a)

    @staticmethod
    def _sample(prob):
        """Bernoulli sample from probability tensor."""
        return (torch.rand_like(prob) < prob).float()

    # --- free energy ---

    def free_energy(self, V):
        """F(v) = -a·v - Σ_j softplus(b_j + W_j·v)"""
        return -(V @ self.a) - F.softplus(V @ self.W + self.b).sum(1)

    # --- reconstruction ---

    @torch.no_grad()
    def reconstruct(self, V):
        """One Gibbs step: V → H → V̂  (returns probabilities, not samples)"""
        ph = self._ph_given_v(V)
        H  = self._sample(ph)
        return self._pv_given_h(H)

    @torch.no_grad()
    def reconstruction_mse(self, V):
        """Mean squared error between V and its one-step reconstruction."""
        return F.mse_loss(self.reconstruct(V), V).item()

    @torch.no_grad()
    def hidden_probs(self, V):
        """Return P(H=1|V) for all samples — used for post-training analysis."""
        return self._ph_given_v(V)

    # --- training ---

    def train(self, X_train, X_val=None,
              epochs=500, lr=0.01, lr_decay=0.998,
              cd_steps=1, batch_i=10, batch_f=256, n_batches=20,
              gamma=1e-4, beta=0.9, epsilon=1e-4,
              eval_every=10, verbose=True):
        """
        CD-k with RMSprop.

        Positive phase:  ph0 = sigmoid(V0 @ W + b)
        Negative phase:  CD-k Gibbs chain from V0
        Gradients:
          dW = (V0.T @ ph0 - Vk.T @ phk) / batch_size
          da = (V0 - Vk).mean(0)
          db = (ph0 - phk).mean(0)
        """
        N          = X_train.shape[0]
        current_lr = lr

        # Hinton (2010) visible bias init from data mean
        xmean  = X_train.mean(0).clamp(1e-4, 1 - 1e-4)
        self.a = torch.log(xmean / (1 - xmean))

        # RMSprop accumulators
        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)

        history = {"train_mse": [], "val_mse": [], "epoch": []}

        pbar = tqdm(range(1, epochs + 1), desc="Training RBM", unit="epoch")
        for epoch in pbar:
            # Linearly anneal batch size
            q          = (epoch - 1) / max(epochs - 1, 1)
            batch_size = int(batch_i + (batch_f - batch_i) * q**2)
            recon_acc  = 0.0

            for _ in range(n_batches):
                idx = torch.randperm(N, device=self.device)[:batch_size]
                V0  = X_train[idx]

                # Positive phase
                ph0 = self._ph_given_v(V0)
                H   = self._sample(ph0)

                # Negative phase (CD-k)
                Vk = V0
                for _ in range(cd_steps):
                    pv  = self._pv_given_h(H)
                    Vk  = self._sample(pv)
                    phk = self._ph_given_v(Vk)
                    H   = self._sample(phk)

                # Gradients
                dW = (V0.t() @ ph0 - Vk.t() @ phk) / batch_size
                da = (V0 - Vk).mean(0)
                db = (ph0 - phk).mean(0)

                recon_acc += F.mse_loss(self._pv_given_h(ph0), V0).item()

                # RMSprop update
                sW = beta * sW + (1 - beta) * dW.pow(2)
                sa = beta * sa + (1 - beta) * da.pow(2)
                sb = beta * sb + (1 - beta) * db.pow(2)

                self.W += current_lr * dW / (sW + epsilon).sqrt()
                self.a += current_lr * da / (sa + epsilon).sqrt()
                self.b += current_lr * db / (sb + epsilon).sqrt()

                # L1 regularisation (encourages sparse hidden activations)
                if gamma > 0:
                    self.W -= gamma * current_lr * self.W.sign()
                    self.a -= gamma * current_lr * self.a.sign()
                    self.b -= gamma * current_lr * self.b.sign()

            current_lr *= lr_decay

            if epoch % eval_every == 0 or epoch == 1:
                train_mse = recon_acc / n_batches
                val_mse   = self.reconstruction_mse(X_val) if X_val is not None else None
                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)

                if verbose:
                    stats = {"train_mse": f"{train_mse:.4f}", "batch": batch_size}
                    if val_mse is not None:
                        stats["val_mse"] = f"{val_mse:.4f}"
                    pbar.set_postfix(stats)

        return history

    def numpy_params(self):
        return (self.W.cpu().float().numpy(),
                self.a.cpu().float().numpy(),
                self.b.cpu().float().numpy())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def export_results_csv(history, W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # CSV 1: training curves
    df_curves = pd.DataFrame({
        "epoch": history["epoch"],
        "train_mse": history["train_mse"]
    })
    if history["val_mse"][0] is not None:
        df_curves["val_mse"] = history["val_mse"]
    path_curves = os.path.join(out_dir, "rbm_training_curves.csv")
    df_curves.to_csv(path_curves, index=False)
    print(f"[CSV]  saved {path_curves}")

    # CSV 2: weight matrix (taxa × hidden units)
    df_weights = pd.DataFrame(W, columns=[f"h{j}" for j in range(W.shape[1])],
                             index=taxa_cols)
    path_weights = os.path.join(out_dir, "rbm_weights.csv")
    df_weights.to_csv(path_weights)
    print(f"[CSV]  saved {path_weights}")


def plot_results(history, W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: reconstruction MSE curves
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_mse"], label="train MSE", color="steelblue")
    if history["val_mse"][0] is not None:
        axes[0].plot(epochs, history["val_mse"], label="val MSE",
                     color="firebrick", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_title("Training curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: weight matrix heatmap (taxa × hidden units)
    im = axes[1].imshow(W, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(W).max(), vmax=np.abs(W).max())
    axes[1].set_xlabel("Hidden unit")
    axes[1].set_ylabel("Taxon")
    axes[1].set_title(f"Weight matrix W  ({W.shape[0]} taxa × {W.shape[1]} hidden)")
    axes[1].set_xticks(range(W.shape[1]))
    axes[1].set_xticklabels([f"h{j}" for j in range(W.shape[1])])

    # Label top-loading taxon per hidden unit
    for j in range(W.shape[1]):
        top_i = np.abs(W[:, j]).argmax()
        axes[1].text(j, top_i, taxa_cols[top_i][:8],
                     ha="center", va="center", fontsize=5, color="black")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    path = os.path.join(out_dir, "rbm_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")


def plot_hidden_activations(rbm, X_train, X_val, dates_train, dates_val, out_dir):
    """
    Plot h(t) = P(H=1 | v_t) over time.
    Key diagnostic: do hidden units show seasonal/annual patterns?
    If h(t) clusters near 0 or 1 → discrete states appropriate.
    If h(t) spreads continuously → Gaussian hidden units warranted.
    """
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        H_train = rbm.hidden_probs(X_train).cpu().numpy()
        H_val   = rbm.hidden_probs(X_val).cpu().numpy()

    H_all     = np.concatenate([H_train, H_val], axis=0)
    dates_all = pd.concat([dates_train, dates_val]).reset_index(drop=True)

    # CSV export: hidden activations over time
    df_hidden = pd.DataFrame(H_all, columns=[f"h{j}" for j in range(H_all.shape[1])])
    df_hidden.insert(0, "date", dates_all.values)
    path_hidden = os.path.join(out_dir, "rbm_hidden_activations.csv")
    df_hidden.to_csv(path_hidden, index=False)
    print(f"[CSV]  saved {path_hidden}")

    n_hidden = H_all.shape[1]
    fig, axes = plt.subplots(n_hidden, 1, figsize=(13, 2.5 * n_hidden), sharex=True)
    if n_hidden == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(dates_all, H_all[:, j], lw=0.6, color="steelblue", alpha=0.7)
        ax.axhline(0.5, color="firebrick", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"P(h{j}=1|v)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Spread diagnostic
        spread = H_all[:, j].std()
        ax.set_title(
            f"Hidden unit {j}  —  std={spread:.3f}  "
            f"({'clusters near 0/1 → discrete ok' if spread < 0.3 else 'spread → consider Gaussian hidden'})",
            fontsize=8
        )

    axes[-1].set_xlabel("Date")
    plt.suptitle("Hidden unit activations h(t) over time", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "hidden_activations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device]  {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[Device]  CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data
    X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds = \
        load_and_binarise(DATA_PATH, binarize=BINARIZE_THRESHOLD,
                          val_frac=VAL_FRAC, device=device)

    print(f"\n[Config]  D={len(taxa_cols)}  L={N_HIDDEN}  "
          f"epochs={EPOCHS}  lr={LR}  cd={CD_STEPS}  "
          f"binarize='{BINARIZE_THRESHOLD}'")

    # Train
    rbm = RBM(n_visible=len(taxa_cols), n_hidden=N_HIDDEN, device=device)
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

    # Save weights
    W, a, b = rbm.numpy_params()
    weights_path = os.path.join(OUTPUT_DIR, "weights.npz")
    np.savez(weights_path, W=W, a=a, b=b,
             taxa=taxa_cols, thresholds=thresholds,
             binarize=BINARIZE_THRESHOLD)
    print(f"[Saved]  weights → {weights_path}")

    # Plots
    if PLOT_RESULTS:
        plot_results(history, W, taxa_cols, OUTPUT_DIR)
        plot_hidden_activations(rbm, X_train, X_val,
                                dates_train, dates_val, OUTPUT_DIR)

    # CSV export
    export_results_csv(history, W, taxa_cols, OUTPUT_DIR)

    print("\n[Final metrics]")
    print(f"  train MSE : {rbm.reconstruction_mse(X_train):.4f}")
    print(f"  val   MSE : {rbm.reconstruction_mse(X_val):.4f}")
    print("\nDone.")