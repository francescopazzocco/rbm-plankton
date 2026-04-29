"""
rbm_plankton.py — RBM on Lake Greifen plankton data
=====================================================

Supports two visible unit models, switchable via VISIBLE_MODEL:

  "bernoulli"  Bernoulli-Bernoulli RBM (current working model)
               Input: binarised counts (v > per-taxon median)
               Validated: seasonal structure recovered, hidden units binary

  "nb"         Negative-Binomial–Bernoulli RBM (experimental)
               Input: raw count concentrations (organisms/μL), no transform
               Rationale: NB is the natural distribution for overdispersed
               count data with structural zeros. Zeros handled by NB likelihood
               directly — no binarisation or log-transform needed.
               Note: data is in organisms/μL (continuous floats, not integers).
               NB log-likelihood is computed via lgamma which supports
               non-integer values. COUNT_SCALE can be used to convert to
               approximate integer counts if desired.

See DECISIONS.md and ARCHITECTURE.md for full rationale.
"""

import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from tqdm import tqdm

# ---------------------------------------------------------------------------
# =====================  CONFIGURABLE PARAMETERS  ===========================
# ---------------------------------------------------------------------------

N_HIDDEN     = 5
EPOCHS       = 500
LR           = 0.01
LR_DECAY     = 0.998
CD_STEPS     = 1
N_BATCHES    = 20
BATCH_I      = 10
BATCH_F      = 256
GAMMA        = 1e-4          # L1 weight regularisation
BETA         = 0.9           # RMSprop momentum
EPSILON      = 1e-4          # RMSprop stability
VAL_FRAC     = 0.15
PLOT_RESULTS = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BRANCH — Visible model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "bernoulli" : Bernoulli-Bernoulli RBM on binarised data  (validated)
# "nb"        : NB-Bernoulli RBM on raw count concentrations (experimental)
#
# See DECISIONS.md for trade-off discussion.

VISIBLE_MODEL = "nb"   # ← SWITCH HERE: "bernoulli" | "nb"

DATA_PATH    = "data/raw/TimeSeries_countsuL_clean.csv"
OUTPUT_DIR   = f"results/{VISIBLE_MODEL}_L{N_HIDDEN}"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BRANCH — Binarisation threshold (Bernoulli model only)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "zero"   : v > 0  (presence/absence)
# "median" : v > per-taxon median  ← CLOSED: median is better
#
BINARIZE_THRESHOLD = "median"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NB-specific parameters
# COUNT_SCALE multiplies raw organisms/μL values before NB modelling.
# The NB log-likelihood is computed via lgamma (supports non-integer values)
# so COUNT_SCALE=1.0 is valid. Increasing it (e.g. 1000) converts the
# concentrations to approximate integer-scale counts if preferred.
COUNT_SCALE    = 1000
THETA_INIT_LOG = 0.0   # initial log(θ) — θ=1 means NB starts like Poisson


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _base_load(path: str, val_frac: float, device: torch.device):
    """Shared steps: parse, sort, drop zero rows, separate NaN rows, split."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa_cols = [c for c in df.columns if c != "date"]

    # Drop all-zero rows (instrument downtime — different data generating process)
    nonzero_mask = df[taxa_cols].sum(axis=1) > 0
    df = df[nonzero_mask].copy()

    # Separate NaN rows → structured test set post-training
    nan_mask = df[taxa_cols].isna().any(axis=1)
    nan_rows = df[nan_mask].copy()
    df       = df[~nan_mask].copy().reset_index(drop=True)

    print(f"[Data]  clean rows: {len(df)}  |  NaN test rows: {len(nan_rows)}")

    n_val   = int(len(df) * val_frac)
    n_train = len(df) - n_val
    dates_train = df["date"].iloc[:n_train]
    dates_val   = df["date"].iloc[n_train:]
    print(f"[Split] train: {n_train}  val: {n_val}  "
          f"(val starts {dates_val.iloc[0].date()})")

    return df, taxa_cols, dates_train, dates_val, nan_rows, n_train


def load_and_binarise(path=DATA_PATH, binarize=BINARIZE_THRESHOLD,
                      val_frac=VAL_FRAC, device=torch.device("cpu")):
    """
    Preprocessing for Bernoulli-Bernoulli RBM.
    Returns binary tensors in {0, 1}.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device)

    X = df[taxa_cols].values.astype(np.float32)

    if binarize == "zero":
        thresholds = np.zeros(len(taxa_cols), dtype=np.float32)
        X_bin = (X > 0).astype(np.float32)
    elif binarize == "median":
        thresholds = np.median(X, axis=0)
        X_bin = (X > thresholds).astype(np.float32)
    else:
        raise ValueError(f"Unknown BINARIZE_THRESHOLD='{binarize}'.")

    n_const = ((X_bin.mean(0) < 0.02) | (X_bin.mean(0) > 0.98)).sum()
    if n_const > 0:
        print(f"[Warning] {n_const} taxa >98% constant after binarisation")
    print(f"[Binarise] threshold='{binarize}'")

    X_train = torch.tensor(X_bin[:n_train],  dtype=torch.float32, device=device)
    X_val   = torch.tensor(X_bin[n_train:],  dtype=torch.float32, device=device)
    return X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds


def load_raw_counts(path=DATA_PATH, scale=COUNT_SCALE,
                    val_frac=VAL_FRAC, device=torch.device("cpu")):
    """
    Preprocessing for NB-Bernoulli RBM.
    Returns raw count concentrations (organisms/μL), optionally scaled.
    No binarisation, no log-transform, no z-score — NB handles the
    distribution directly.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device)

    X = df[taxa_cols].fillna(0).values.astype(np.float32) * scale

    print(f"[Counts] scale={scale}  range=[{X.min():.4f}, {X.max():.4f}]  "
          f"zeros={( X==0).mean():.1%}")

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val   = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    return X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows


# ---------------------------------------------------------------------------
# Bernoulli-Bernoulli RBM  (validated, unchanged)
# ---------------------------------------------------------------------------

class RBM:
    """
    Bernoulli-Bernoulli RBM trained by CD-k with RMSprop.

    Validated on plankton data (L=5, median threshold):
      - All 5 hidden units empirically binary (>90% near 0 or 1)
      - Seasonal community structure recovered without temporal supervision
      - Ecologically coherent taxa groupings per hidden unit
    """

    def __init__(self, n_visible: int, n_hidden: int,
                 device=torch.device("cpu")):
        self.D      = n_visible
        self.L      = n_hidden
        self.device = device

        scale  = math.sqrt(4.0 / (n_visible + n_hidden))
        self.W = torch.randn(n_visible, n_hidden, device=device) * scale
        self.b = torch.zeros(n_hidden,  device=device)
        self.a = torch.zeros(n_visible, device=device)

    def _ph_given_v(self, V):
        return torch.sigmoid(V @ self.W + self.b)

    def _pv_given_h(self, H):
        return torch.sigmoid(H @ self.W.t() + self.a)

    @staticmethod
    def _sample(prob):
        return (torch.rand_like(prob) < prob).float()

    def free_energy(self, V):
        return -(V @ self.a) - F.softplus(V @ self.W + self.b).sum(1)

    @torch.no_grad()
    def reconstruct(self, V):
        H = self._sample(self._ph_given_v(V))
        return self._pv_given_h(H)

    @torch.no_grad()
    def reconstruction_mse(self, V):
        return F.mse_loss(self.reconstruct(V), V).item()

    @torch.no_grad()
    def hidden_probs(self, V):
        return self._ph_given_v(V)

    def train(self, X_train, X_val=None,
              epochs=500, lr=0.01, lr_decay=0.998,
              cd_steps=1, batch_i=10, batch_f=256, n_batches=20,
              gamma=1e-4, beta=0.9, epsilon=1e-4,
              eval_every=10, verbose=True):

        N          = X_train.shape[0]
        current_lr = lr

        xmean  = X_train.mean(0).clamp(1e-4, 1 - 1e-4)
        self.a = torch.log(xmean / (1 - xmean))

        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)

        history = {"train_mse": [], "val_mse": [], "epoch": []}

        pbar = tqdm(range(1, epochs + 1), desc="Training RBM [Bernoulli]",
                    unit="epoch")
        for epoch in pbar:
            q          = (epoch - 1) / max(epochs - 1, 1)
            batch_size = int(batch_i + (batch_f - batch_i) * q**2)
            recon_acc  = 0.0

            for _ in range(n_batches):
                idx = torch.randperm(N, device=self.device)[:batch_size]
                V0  = X_train[idx]

                ph0 = self._ph_given_v(V0)
                H   = self._sample(ph0)

                Vk = V0
                for _ in range(cd_steps):
                    pv  = self._pv_given_h(H)
                    Vk  = self._sample(pv)
                    phk = self._ph_given_v(Vk)
                    H   = self._sample(phk)

                dW = (V0.t() @ ph0 - Vk.t() @ phk) / batch_size
                da = (V0 - Vk).mean(0)
                db = (ph0 - phk).mean(0)

                recon_acc += F.mse_loss(self._pv_given_h(ph0), V0).item()

                sW = beta * sW + (1 - beta) * dW.pow(2)
                sa = beta * sa + (1 - beta) * da.pow(2)
                sb = beta * sb + (1 - beta) * db.pow(2)

                self.W += current_lr * dW / (sW + epsilon).sqrt()
                self.a += current_lr * da / (sa + epsilon).sqrt()
                self.b += current_lr * db / (sb + epsilon).sqrt()

                if gamma > 0:
                    self.W -= gamma * current_lr * self.W.sign()
                    self.a -= gamma * current_lr * self.a.sign()
                    self.b -= gamma * current_lr * self.b.sign()

            current_lr *= lr_decay

            if epoch % eval_every == 0 or epoch == 1:
                train_mse = recon_acc / n_batches
                val_mse   = self.reconstruction_mse(X_val) \
                            if X_val is not None else None
                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                if verbose:
                    stats = {"train_mse": f"{train_mse:.4f}",
                             "batch": batch_size}
                    if val_mse is not None:
                        stats["val_mse"] = f"{val_mse:.4f}"
                    pbar.set_postfix(stats)

        return history

    def numpy_params(self):
        return (self.W.cpu().float().numpy(),
                self.a.cpu().float().numpy(),
                self.b.cpu().float().numpy())


# ---------------------------------------------------------------------------
# Negative-Binomial–Bernoulli RBM  (experimental)
# ---------------------------------------------------------------------------

class NBRBM:
    """
    NB-Bernoulli RBM trained by CD-k with RMSprop.

    Visible units: Negative Binomial
      p(v_i | h) = NB(μ_i, θ_i)
      μ_i = exp(a_i + Σ_j W_ij * h_j)   ← exp ensures positivity
      θ_i = exp(log_theta_i)             ← learned per-taxon dispersion

    Hidden units: Bernoulli (unchanged from standard RBM)
      p(h_j=1 | v) = σ(b_j + Σ_i W_ij * v_i)

    NB log-likelihood (continuous support via lgamma):
      log NB(v; μ, θ) = lgamma(v+θ) - lgamma(θ) - lgamma(v+1)
                       + θ*log(θ/(θ+μ)) + v*log(μ/(θ+μ))

    CD-k gradient derivation:
      ∂log NB(v_i|h)/∂η_i  = θ_i*(v_i - μ_i) / (μ_i + θ_i)
      where η_i = a_i + W_i·h  and  ∂μ_i/∂η_i = μ_i

      → dW_ij ∝ Σ_batch [h_j^+ * r_i^+  -  h_j^- * r_i^-]
      → da_i  ∝ Σ_batch [r_i^+ - r_i^-]
      where r_i = θ_i*(v_i - μ_i)/(μ_i + θ_i)   (weighted residual)

      db_j uses standard CD: ph0_j - phk_j  (hidden bias unaffected by
      visible distribution)

      θ update: gradient of NB log-likelihood w.r.t. log_theta_i,
      computed via autograd on the positive phase batch only.
      (θ does not participate in the CD chain — only in the likelihood.)

    Data note:
      The CSV reports organisms/μL (continuous floats in [0, 0.44]).
      NB is defined on non-negative integers but the lgamma formulation
      extends it to non-negative reals. COUNT_SCALE can be used to
      convert concentrations to approximate integer-scale counts.
    """

    def __init__(self, n_visible: int, n_hidden: int,
                 device=torch.device("cpu"),
                 theta_init_log: float = THETA_INIT_LOG):
        self.D      = n_visible
        self.L      = n_hidden
        self.device = device

        scale  = math.sqrt(4.0 / (n_visible + n_hidden))
        self.W = torch.randn(n_visible, n_hidden, device=device) * scale
        self.b = torch.zeros(n_hidden,  device=device)
        # a is the log-mean baseline: μ_i = exp(a_i + W_i·h)
        # initialise a_i = log(mean(v_i) + ε) so μ at h=0 matches data mean
        self.a = torch.zeros(n_visible, device=device)

        # log_theta: one per taxon, learnable
        # θ = exp(log_theta);  init log_theta=0 → θ=1 (moderate overdispersion)
        self.log_theta = torch.full((n_visible,), theta_init_log,
                                    device=device, requires_grad=True)

    # --- internal helpers ---

    def _eta(self, H):
        """Linear predictor: η = a + H @ W.T  →  shape (batch, D)"""
        return self.a.unsqueeze(0) + H @ self.W.t()

    def _mu(self, H):
        """NB mean: μ_i = exp(η_i),  shape (batch, D)"""
        return torch.exp(self._eta(H))

    def _ph_given_v(self, V):
        """P(H=1|V) = σ(b + V @ W),  shape (batch, L)"""
        return torch.sigmoid(V @ self.W + self.b)

    @staticmethod
    def _sample_bernoulli(prob):
        return (torch.rand_like(prob) < prob).float()

    def _sample_nb(self, mu):
        """
        Sample from NB(μ, θ) using the Gamma-Poisson mixture:
          g ~ Gamma(θ, θ/μ)   →   v ~ Poisson(g)
        Returns float tensor (Poisson samples are non-negative integers).
        """
        theta = self.log_theta.detach().exp().clamp(min=1e-4)
        # Gamma(concentration=θ, rate=θ/μ) = Gamma(θ, θ/μ)
        concentration = theta.unsqueeze(0).expand_as(mu)
        rate          = (theta / mu.clamp(min=1e-8)).unsqueeze(0) \
                        if theta.dim() == 1 else theta / mu.clamp(min=1e-8)
        rate          = theta.unsqueeze(0) / mu.clamp(min=1e-8)
        g  = torch.distributions.Gamma(concentration, rate).sample()
        v  = torch.poisson(g)
        return v.float()

    def _nb_log_prob(self, V, mu):
        """
        NB log-likelihood via lgamma — supports non-integer V.
        log NB(v; μ, θ) = lgamma(v+θ) - lgamma(θ) - lgamma(v+1)
                         + θ*log(θ/(θ+μ)) + v*log(μ/(θ+μ))
        Shape: (batch, D) → scalar (mean over batch and taxa)
        """
        theta = self.log_theta.exp().clamp(min=1e-4)
        eps   = 1e-8

        log_nb = (torch.lgamma(V + theta)
                  - torch.lgamma(theta)
                  - torch.lgamma(V + 1)
                  + theta * torch.log(theta / (theta + mu + eps))
                  + V     * torch.log(mu    / (theta + mu + eps)))
        return log_nb.mean()

    def _nb_residual(self, V, mu):
        """
        Weighted residual r_i = θ_i*(v_i - μ_i)/(μ_i + θ_i)
        This is ∂log NB(v_i|h)/∂η_i — used in CD gradients for W and a.
        Shape: (batch, D)
        """
        theta = self.log_theta.detach().exp().clamp(min=1e-4)
        return theta * (V - mu) / (mu + theta + 1e-8)

    # --- public interface ---

    @torch.no_grad()
    def reconstruct(self, V):
        """V → h sample → μ (NB mean, not a sample)"""
        ph = self._ph_given_v(V)
        H  = self._sample_bernoulli(ph)
        return self._mu(H)

    @torch.no_grad()
    def reconstruction_mse(self, V):
        return F.mse_loss(self.reconstruct(V), V).item()

    @torch.no_grad()
    def hidden_probs(self, V):
        return self._ph_given_v(V)

    def nll(self, V):
        """Negative log-likelihood on V (positive phase only, no CD)."""
        with torch.no_grad():
            ph = self._ph_given_v(V)
            H  = self._sample_bernoulli(ph)
        mu = self._mu(H)
        return -self._nb_log_prob(V, mu).item()

    def train(self, X_train, X_val=None,
              epochs=500, lr=0.01, lr_decay=0.998,
              cd_steps=1, batch_i=10, batch_f=256, n_batches=20,
              gamma=1e-4, beta=0.9, epsilon=1e-4,
              lr_theta=None,
              eval_every=10, verbose=True):
        """
        CD-k training for NB-Bernoulli RBM.

        Parameters
        ----------
        lr_theta : float | None
            Learning rate for θ (dispersion). If None, uses lr * 0.1.
            θ is updated via autograd on the positive phase NB log-likelihood
            (not via CD — θ does not affect the Gibbs chain direction).
        """
        N           = X_train.shape[0]
        current_lr  = lr
        lr_theta    = lr_theta or lr * 0.1

        # Initialise a to log(mean counts + ε) so μ at h=0 ≈ data mean
        data_mean = X_train.mean(0).clamp(min=1e-8)
        self.a    = torch.log(data_mean)

        # RMSprop accumulators for W, a, b
        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)
        # Adam-style accumulator for log_theta
        s_theta = torch.zeros_like(self.log_theta.data)

        history = {"train_mse": [], "val_mse": [], "train_nll": [],
                   "val_nll": [], "theta_mean": [],
                   "sat_lo": [], "sat_hi": [], "sat_mid": [], "epoch": []}

        pbar = tqdm(range(1, epochs + 1), desc="Training RBM [NB]",
                    unit="epoch")
        for epoch in pbar:
            q          = (epoch - 1) / max(epochs - 1, 1)
            batch_size = int(batch_i + (batch_f - batch_i) * q**2)
            recon_acc  = 0.0

            for _ in range(n_batches):
                idx = torch.randperm(N, device=self.device)[:batch_size]
                V0  = X_train[idx]

                # ── Positive phase ────────────────────────────────────────
                ph0 = self._ph_given_v(V0)
                H0  = self._sample_bernoulli(ph0)
                mu0 = self._mu(H0)
                r0  = self._nb_residual(V0, mu0)       # (batch, D)

                # ── Negative phase (CD-k) ─────────────────────────────────
                Hk = H0
                for _ in range(cd_steps):
                    Vk  = self._sample_nb(self._mu(Hk))
                    phk = self._ph_given_v(Vk)
                    Hk  = self._sample_bernoulli(phk)

                muk = self._mu(Hk)
                rk  = self._nb_residual(Vk, muk)       # (batch, D)

                # ── CD gradients for W, a, b ──────────────────────────────
                # dW: outer product of residuals and hidden probs
                dW = (r0.t() @ ph0  - rk.t() @ phk) / batch_size
                da = (r0 - rk).mean(0)
                db = (ph0 - phk).mean(0)

                recon_acc += F.mse_loss(mu0, V0).item()

                # RMSprop update for W, a, b
                sW = beta * sW + (1 - beta) * dW.pow(2)
                sa = beta * sa + (1 - beta) * da.pow(2)
                sb = beta * sb + (1 - beta) * db.pow(2)

                self.W += current_lr * dW / (sW + epsilon).sqrt()
                self.a += current_lr * da / (sa + epsilon).sqrt()
                self.b += current_lr * db / (sb + epsilon).sqrt()

                if gamma > 0:
                    self.W -= gamma * current_lr * self.W.sign()

                # ── θ update via autograd on positive phase NLL ───────────
                # Re-enable grad for log_theta; keep other params detached
                mu0_for_theta = torch.exp(
                    self.a.detach() + H0.detach() @ self.W.detach().t()
                )
                # Temporarily enable grad on log_theta
                self.log_theta.requires_grad_(True)
                theta  = self.log_theta.exp().clamp(min=1e-4)
                eps_nb = 1e-8
                log_nb = (torch.lgamma(V0 + theta)
                          - torch.lgamma(theta)
                          - torch.lgamma(V0 + 1)
                          + theta * torch.log(theta / (theta + mu0_for_theta + eps_nb))
                          + V0    * torch.log(mu0_for_theta / (theta + mu0_for_theta + eps_nb)))
                nll_theta = -log_nb.mean()
                nll_theta.backward()

                with torch.no_grad():
                    g_theta = self.log_theta.grad.clone()
                    s_theta = beta * s_theta + (1 - beta) * g_theta.pow(2)
                    self.log_theta -= lr_theta * g_theta / (s_theta + epsilon).sqrt()
                    self.log_theta.grad.zero_()
                self.log_theta.requires_grad_(False)

            current_lr *= lr_decay

            if epoch % eval_every == 0 or epoch == 1:
                train_mse  = recon_acc / n_batches
                val_mse    = self.reconstruction_mse(X_val) \
                             if X_val is not None else None
                train_nll  = self.nll(X_train)
                val_nll    = self.nll(X_val) if X_val is not None else None
                theta_mean = self.log_theta.detach().exp().mean().item()

                with torch.no_grad():
                    ph = self._ph_given_v(X_train)
                sat_lo  = (ph < 0.1).float().mean().item()
                sat_hi  = (ph > 0.9).float().mean().item()
                sat_mid = 1.0 - sat_lo - sat_hi

                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                history["train_nll"].append(train_nll)
                history["val_nll"].append(val_nll)
                history["theta_mean"].append(theta_mean)
                history["sat_lo"].append(sat_lo)
                history["sat_hi"].append(sat_hi)
                history["sat_mid"].append(sat_mid)

                if verbose:
                    stats = {"nll": f"{train_nll:.2f}",
                             "θ_mean": f"{theta_mean:.3f}",
                             "sat_mid": f"{sat_mid:.0%}",
                             "batch": batch_size}
                    if val_nll is not None:
                        stats["val_nll"] = f"{val_nll:.2f}"
                    pbar.set_postfix(stats)

        return history

    def numpy_params(self):
        return (self.W.cpu().float().numpy(),
                self.a.cpu().float().numpy(),
                self.b.cpu().float().numpy(),
                self.log_theta.detach().cpu().float().numpy())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def export_results_csv(history, W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cols = {"epoch": history["epoch"], "train_mse": history["train_mse"]}
    if history.get("val_mse") and history["val_mse"][0] is not None:
        cols["val_mse"] = history["val_mse"]
    if history.get("train_nll"):
        cols["train_nll"] = history["train_nll"]
    if history.get("val_nll") and history["val_nll"][0] is not None:
        cols["val_nll"] = history["val_nll"]
    if history.get("theta_mean"):
        cols["theta_mean"] = history["theta_mean"]
    if history.get("sat_mid"):
        cols["sat_lo"]  = history["sat_lo"]
        cols["sat_hi"]  = history["sat_hi"]
        cols["sat_mid"] = history["sat_mid"]

    pd.DataFrame(cols).to_csv(
        os.path.join(out_dir, "rbm_training_curves.csv"), index=False)

    pd.DataFrame(W, columns=[f"h{j}" for j in range(W.shape[1])],
                 index=taxa_cols).to_csv(
        os.path.join(out_dir, "rbm_weights.csv"))

    print(f"[CSV]  saved training curves and weights → {out_dir}/")


def plot_results(history, W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: training curves
    ax = axes[0]
    ax.plot(history["epoch"], history["train_mse"],
            color="steelblue", lw=1.5, label="train MSE")
    if history.get("val_mse") and history["val_mse"][0] is not None:
        ax.plot(history["epoch"], history["val_mse"],
                color="firebrick", lw=1.5, ls="--", label="val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Training curves"); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: weight heatmap (clustered)
    link  = linkage(W, method="ward")
    order = leaves_list(link)
    W_ord = W[order]
    vmax  = np.abs(W).max()
    im    = axes[1].imshow(W_ord, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
    axes[1].set_xticks(range(W.shape[1]))
    axes[1].set_xticklabels([f"h{j}" for j in range(W.shape[1])], fontsize=9)
    axes[1].set_yticks(range(len(W_ord)))
    axes[1].set_yticklabels([taxa_cols[i] for i in order], fontsize=4)
    axes[1].set_title("Weight matrix W (taxa clustered)")
    plt.colorbar(im, ax=axes[1], shrink=0.6)

    plt.tight_layout()
    path = os.path.join(out_dir, "rbm_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")


def plot_hidden_activations(rbm, X_train, X_val,
                            dates_train, dates_val, out_dir):
    import matplotlib.dates as mdates
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        H_all  = torch.cat([rbm.hidden_probs(X_train),
                             rbm.hidden_probs(X_val)], dim=0).cpu().numpy()
    dates_all = pd.concat([dates_train, dates_val]).reset_index(drop=True)

    n_hidden = H_all.shape[1]
    colors   = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    fig, axes = plt.subplots(n_hidden, 1,
                              figsize=(13, 2.5 * n_hidden), sharex=True)
    if n_hidden == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        vals   = H_all[:, j]
        near_0 = (vals < 0.1).mean()
        near_1 = (vals > 0.9).mean()
        mid    = 1 - near_0 - near_1

        ax.plot(dates_all, vals, lw=0.5, color=colors[j % len(colors)],
                alpha=0.6)
        ax.plot(dates_all,
                pd.Series(vals).rolling(14, center=True).mean(),
                lw=1.8, color=colors[j % len(colors)])
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(h=1|v)", fontsize=8)
        ax.set_title(
            f"h{j}  |  <0.1: {near_0:.0%}   >0.9: {near_1:.0%}   "
            f"middle: {mid:.0%}  →  "
            f"{'binary ✓' if mid < 0.15 else 'continuous'}",
            fontsize=8, loc="left"
        )
        ax.grid(True, alpha=0.25)
        for year in range(2019, 2025):
            ax.axvspan(pd.Timestamp(f"{year}-06-01"),
                       pd.Timestamp(f"{year}-09-01"),
                       alpha=0.07, color="orange")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.suptitle("Hidden unit activations h(t)  |  orange = summer",
                 fontsize=11)
    plt.tight_layout()

    path = os.path.join(out_dir, "hidden_activations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")

    # CSV export for external analysis
    df_h = pd.DataFrame(H_all, columns=[f"h{j}" for j in range(n_hidden)])
    df_h.insert(0, "date", dates_all.values)
    df_h.to_csv(os.path.join(out_dir, "rbm_hidden_activations.csv"),
                index=False)
    print(f"[CSV]  saved hidden activations → {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device]  {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[Config]  VISIBLE_MODEL='{VISIBLE_MODEL}'  L={N_HIDDEN}  "
          f"epochs={EPOCHS}  lr={LR}  cd={CD_STEPS}")

    # ── Load data ────────────────────────────────────────────────────────
    if VISIBLE_MODEL == "bernoulli":
        X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds = \
            load_and_binarise(DATA_PATH, binarize=BINARIZE_THRESHOLD,
                              val_frac=VAL_FRAC, device=device)
        rbm = RBM(n_visible=len(taxa_cols), n_hidden=N_HIDDEN, device=device)

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

    # ── Train ────────────────────────────────────────────────────────────
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

    # ── Save ─────────────────────────────────────────────────────────────
    params = rbm.numpy_params()
    W, a, b = params[0], params[1], params[2]

    save_dict = dict(W=W, a=a, b=b, taxa=taxa_cols,
                     visible_model=VISIBLE_MODEL)
    if VISIBLE_MODEL == "nb":
        save_dict["log_theta"] = params[3]
    if thresholds is not None:
        save_dict["thresholds"] = thresholds

    weights_path = os.path.join(OUTPUT_DIR, "weights.npz")
    np.savez(weights_path, **save_dict)
    print(f"[Saved]  weights → {weights_path}")

    # ── Plots and CSV export ─────────────────────────────────────────────
    if PLOT_RESULTS:
        plot_results(history, W, taxa_cols, OUTPUT_DIR)
        plot_hidden_activations(rbm, X_train, X_val,
                                dates_train, dates_val, OUTPUT_DIR)
    export_results_csv(history, W, taxa_cols, OUTPUT_DIR)

    print("\n[Final metrics]")
    print(f"  train MSE : {rbm.reconstruction_mse(X_train):.4f}")
    print(f"  val   MSE : {rbm.reconstruction_mse(X_val):.4f}")
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