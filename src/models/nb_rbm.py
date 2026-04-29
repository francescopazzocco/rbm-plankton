"""
nb_rbm.py — Negative-Binomial–Bernoulli RBM (experimental)
=======================================================
NB visible units with Bernoulli hidden units.
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_rbm import BaseRBM


class NBRBM(BaseRBM):
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
    """

    def __init__(self, n_visible, n_hidden,
                 device=torch.device("cpu"),
                 theta_init_log=0.0):
        super().__init__(n_visible, n_hidden, device, scale_init=True)
        self.log_theta = torch.full((n_visible,), theta_init_log,
                                    device=device, requires_grad=False)

    # --- internal helpers ---

    def _eta(self, H):
        """Linear predictor: η = a + H @ W.T  →  shape (batch, D)"""
        return self.a.unsqueeze(0) + H @ self.W.t()

    def _mu(self, H):
        """NB mean: μ_i = exp(η_i), clamped to prevent float32 overflow."""
        return torch.exp(self._eta(H).clamp(max=10.0))

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
        concentration = theta.unsqueeze(0).expand_as(mu)
        rate = theta.unsqueeze(0) / mu.clamp(min=1e-8)
        g = torch.distributions.Gamma(concentration, rate).sample()
        v = torch.poisson(g)
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

        data_mean = X_train.mean(0).clamp(min=1e-8)
        self.a    = torch.log(data_mean)

        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)
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

                ph0 = self._ph_given_v(V0)
                H0  = self._sample_bernoulli(ph0)
                mu0 = self._mu(H0)
                r0  = self._nb_residual(V0, mu0)

                Hk = H0
                for _ in range(cd_steps):
                    Vk  = self._sample_nb(self._mu(Hk))
                    phk = self._ph_given_v(Vk)
                    Hk  = self._sample_bernoulli(phk)

                muk = self._mu(Hk)
                rk  = self._nb_residual(Vk, muk)

                dW = (r0.t() @ ph0  - rk.t() @ phk) / batch_size
                da = (r0 - rk).mean(0)
                db = (ph0 - phk).mean(0)

                recon_acc += F.mse_loss(mu0, V0).item()

                sW = beta * sW + (1 - beta) * dW.pow(2)
                sa = beta * sa + (1 - beta) * da.pow(2)
                sb = beta * sb + (1 - beta) * db.pow(2)

                self.W += current_lr * dW / (sW + epsilon).sqrt()
                self.a += current_lr * da / (sa + epsilon).sqrt()
                self.b += current_lr * db / (sb + epsilon).sqrt()

                if gamma > 0:
                    self.W -= gamma * current_lr * self.W.sign()

                # θ update via autograd on positive phase NLL
                self.log_theta.requires_grad_(True)
                mu0_for_theta = self._mu(H0.detach())
                nll_theta = -self._nb_log_prob(V0, mu0_for_theta)
                nll_theta.backward()

                with torch.no_grad():
                    g_theta = self.log_theta.grad.nan_to_num(nan=0.0).clone()
                    s_theta = beta * s_theta + (1 - beta) * g_theta.pow(2)
                    self.log_theta -= lr_theta * g_theta / (s_theta + epsilon).sqrt()
                    self.log_theta.clamp_(-10.0, 10.0)
                    self.log_theta.grad.zero_()
                self.log_theta.requires_grad_(False)

            current_lr *= lr_decay

            if epoch % eval_every == 0 or epoch == 1:
                train_mse   = recon_acc / n_batches
                val_mse     = self.reconstruction_mse(X_val) \
                              if X_val is not None else None
                train_nll   = self.nll(X_train)
                val_nll     = self.nll(X_val) if X_val is not None else None
                theta_mean  = self.log_theta.detach().exp().mean().item()

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
