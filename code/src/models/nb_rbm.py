"""
nb_rbm.py - Negative-Binomial-Bernoulli RBM
============================================
NB visible units with Bernoulli hidden units. Canonical model family (L=6 selected).
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_rbm import BaseRBM
from ._hidden_monitors import BernoulliHiddenMonitor, ReLUHiddenMonitor, SigmoidHiddenMonitor, SoftmaxHiddenMonitor
from ._constants import (
    THETA_CLAMP_MIN, LOG_PROB_EPS, ETA_CLAMP_MAX,
    LOG_PARAM_CLAMP_MIN, LOG_PARAM_CLAMP_MAX,
    RELU_HIDDEN_CLAMP_MAX, LR_PARAM_MULTIPLIER,
    DEFAULT_LR_DECAY, DEFAULT_GAMMA,
    DEFAULT_BATCH_I, DEFAULT_BATCH_F, DEFAULT_N_BATCHES,
    DEFAULT_BETA, DEFAULT_RMSPROP_EPS,
)


class NB_RBM(BernoulliHiddenMonitor, BaseRBM):
    """
    NB-Bernoulli RBM trained by CD-k with RMSprop.

    Visible units: Negative Binomial
      p(v_i | h) = NB(u_i, theta_i)
      u_i = exp(a_i + sum_j W_ij * h_j)      <- exp ensures positivity
      theta_i = exp(log_theta_i)             <- learned per-taxon dispersion

    Hidden units: Bernoulli (unchanged from standard RBM)
      p(h_j=1 | v) = sum(b_j + sum_i W_ij * v_i)

    NB log-likelihood (continuous support via lgamma):
      log NB(v; u, theta) = lgamma(v+theta) - lgamma(theta) - lgamma(v+1)
                       + theta*log(theta/(theta+u)) + v*log(u/(theta+u))

    CD-k gradient derivation:
      dlog NB(v_i|h)/dn_i  = theta_i*(v_i - u_i) / (u_i + theta_i)
      where n_i = a_i + W_i*h  and  du_i/dn_i = u_i

      -> dW_ij ~ sum_batch [h_j^+ * r_i^+  -  h_j^- * r_i^-]
      -> da_i  ~ sum_batch [r_i^+ - r_i^-]
      where r_i = theta_i*(v_i - u_i)/(u_i + theta_i)   (weighted residual)

      db_j uses standard CD: ph0_j - phk_j  (hidden bias unaffected by
      visible distribution)

      theta update: gradient of NB log-likelihood w.r.t. log_theta_i,
      computed via autograd on the positive phase batch only.
      (theta does not participate in the CD chain - only in the likelihood.)
    """

    def __init__(self, n_visible, n_hidden,
                 device=torch.device("cpu"),
                 theta_init_log=0.0):
        super().__init__(n_visible, n_hidden, device, scale_init=True)
        self.log_theta = torch.full((n_visible,), theta_init_log,
                                    device=device, requires_grad=False)

    # --- internal helpers ---

    def _eta(self, H):
        """Linear predictor: n = a + H @ W.T  ->  shape (batch, D)"""
        return self.a.unsqueeze(0) + H @ self.W.t()

    def _mu(self, H):
        """NB mean: u_i = exp(n_i), clamped to prevent float32 overflow."""
        return torch.exp(self._eta(H).clamp(max=ETA_CLAMP_MAX))

    def _ph_given_v(self, V):
        """P(H=1|V) = sum(b + V @ W),  shape (batch, L)"""
        return torch.sigmoid(V @ self.W + self.b)

    @staticmethod
    def _sample_bernoulli(prob):
        return (torch.rand_like(prob) < prob).float()

    def _sample_nb(self, mu):
        """
        Sample from NB(u, theta) using the Gamma-Poisson mixture:
          g ~ Gamma(theta, theta/u)   ->   v ~ Poisson(g)
        Returns float tensor (Poisson samples are non-negative integers).
        """
        theta = self.log_theta.detach().exp().clamp(min=THETA_CLAMP_MIN)
        concentration = theta.unsqueeze(0).expand_as(mu)
        rate = theta.unsqueeze(0) / mu.clamp(min=LOG_PROB_EPS)
        g = torch.distributions.Gamma(concentration, rate).sample()
        v = torch.poisson(g)
        return v.float()

    def _nb_log_prob(self, V, mu):
        """
        NB log-likelihood via lgamma - supports non-integer V.
        log NB(v; u, theta) = lgamma(v+theta) - lgamma(theta) - lgamma(v+1)
                         + theta*log(theta/(theta+u)) + v*log(u/(theta+u))
        Shape: (batch, D) -> scalar (mean over batch and taxa)
        """
        theta = self.log_theta.exp().clamp(min=THETA_CLAMP_MIN)
        eps   = LOG_PROB_EPS

        log_nb = (torch.lgamma(V + theta)
                  - torch.lgamma(theta)
                  - torch.lgamma(V + 1)
                  + theta * torch.log(theta / (theta + mu + eps))
                  + V     * torch.log(mu    / (theta + mu + eps)))
        return log_nb.mean()

    def _nb_residual(self, V, mu):
        """
        Weighted residual r_i = theta_i*(v_i - u_i)/(u_i + theta_i)
        This is dlog NB(v_i|h)/dn_i - used in CD gradients for W and a.
        Shape: (batch, D)
        """
        theta = self.log_theta.detach().exp().clamp(min=THETA_CLAMP_MIN)
        return theta * (V - mu) / (mu + theta + LOG_PROB_EPS)

    # --- public interface ---

    @torch.no_grad()
    def reconstruct(self, V):
        """V -> h sample -> u (NB mean, not a sample)"""
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
              epochs=500, lr=0.01, lr_decay=DEFAULT_LR_DECAY,
              cd_steps=1, batch_i=DEFAULT_BATCH_I, batch_f=DEFAULT_BATCH_F, n_batches=DEFAULT_N_BATCHES,
              gamma=DEFAULT_GAMMA, beta=DEFAULT_BETA, epsilon=DEFAULT_RMSPROP_EPS,
              lr_theta=None,
              use_pcd=False, n_pcd_chains=500,
              eval_every=10, verbose=True):
        """
        CD-k / PCD-k training for NB-Bernoulli RBM.

        Parameters
        ----------
        lr_theta : float | None
            Learning rate for theta (dispersion). If None, uses lr * 0.1.
            theta is updated via autograd on the positive phase NB log-likelihood
            (not via CD - theta does not affect the Gibbs chain direction).
        use_pcd : bool
            Use Persistent CD instead of standard CD. Persistent chains are
            maintained across batches so they can cross energy barriers between
            modes - the structural fix for slow mixing at L>=5.
        n_pcd_chains : int
            Number of persistent fantasy particles. Must be >= batch_f.
        """
        N           = X_train.shape[0]
        current_lr  = lr
        lr_theta    = lr_theta or lr * LR_PARAM_MULTIPLIER

        data_mean = X_train.mean(0).clamp(min=LOG_PROB_EPS)
        self.a    = torch.log(data_mean)

        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)
        s_theta = torch.zeros_like(self.log_theta.data)

        history = {"train_mse": [], "val_mse": [], "train_nll": [],
                   "val_nll": [], "theta_mean": [], "epoch": []}
        history.update(self._hidden_stats_init())

        # PCD: initialise persistent particle buffer from training data
        if use_pcd:
            pcd_init = torch.randperm(N, device=self.device)[:n_pcd_chains]
            V_pcd = X_train[pcd_init].clone()

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

                if use_pcd:
                    # Negative phase: continue persistent chains
                    sel = torch.randint(0, n_pcd_chains, (batch_size,),
                                        device=self.device)
                    Vk = V_pcd[sel]
                    for _ in range(cd_steps):
                        phk = self._ph_given_v(Vk)
                        Hk  = self._sample_bernoulli(phk)
                        Vk  = self._sample_nb(self._mu(Hk))
                    V_pcd[sel] = Vk.detach()   # persist the new state
                    phk = self._ph_given_v(Vk)
                    Hk  = self._sample_bernoulli(phk)
                else:
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

                # theta update via autograd on positive phase NLL
                self.log_theta.requires_grad_(True)
                mu0_for_theta = self._mu(H0.detach())
                nll_theta = -self._nb_log_prob(V0, mu0_for_theta)
                nll_theta.backward()

                with torch.no_grad():
                    g_theta = self.log_theta.grad.nan_to_num(nan=0.0).clone()
                    s_theta = beta * s_theta + (1 - beta) * g_theta.pow(2)
                    self.log_theta -= lr_theta * g_theta / (s_theta + epsilon).sqrt()
                    self.log_theta.clamp_(LOG_PARAM_CLAMP_MIN, LOG_PARAM_CLAMP_MAX)
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

                hid_stats = self._compute_hidden_stats(X_train)

                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                history["train_nll"].append(train_nll)
                history["val_nll"].append(val_nll)
                history["theta_mean"].append(theta_mean)
                for k, v in hid_stats.items():
                    history[k].append(v)

                if verbose:
                    stats = {"nll": f"{train_nll:.2f}",
                             "theta_mean": f"{theta_mean:.3f}",
                             "batch": batch_size}
                    stats.update(self._hidden_stats_display(hid_stats))
                    if val_nll is not None:
                        stats["val_nll"] = f"{val_nll:.2f}"
                    pbar.set_postfix(stats)

        return history

    def numpy_params(self):
        return (self.W.cpu().float().numpy(),
                self.a.cpu().float().numpy(),
                self.b.cpu().float().numpy(),
                self.log_theta.detach().cpu().float().numpy())


class NB_ReLU_RBM(ReLUHiddenMonitor, NB_RBM):
    """
    NB-ReLU RBM: NB visible units with Rectified Gaussian hidden units.

    Hidden units:
      pre_j = b_j + sum_i W_ij * v_i          (raw pre-activation, can be < 0)
      h_j   ~ clamp(max(0, pre_j + N(0,1)), 0, 5)

    Sampling note: _ph_given_v returns the raw pre-activation (not relu'd) so
    that _sample_hidden adds noise before applying relu — the correct truncated
    Gaussian form. The upper clamp [0, 5] is required because count-scale visible
    units (V up to 444) push pre-activations into the hundreds, overwhelming the
    h^2/2 restoring force from the Gaussian-ReLU energy (which assumes normalised
    inputs). The clamp enforces the energy ceiling externally.
    """

    def _ph_given_v(self, V):
        """Raw pre-activation b + V@W — not relu'd. Sampling needs the signed value."""
        return V @ self.W + self.b

    def _sample_hidden(self, pre_act):
        return F.relu(pre_act + torch.randn_like(pre_act)).clamp(max=RELU_HIDDEN_CLAMP_MAX)

    def _sample_bernoulli(self, prob):
        """Overridden: rectified Gaussian sampling in place of Bernoulli."""
        return self._sample_hidden(prob)

    def hidden_probs(self, V):
        """Approximate E[h|v] = clamp(relu(pre_activation), 0, 5)."""
        return F.relu(self._ph_given_v(V)).clamp(max=RELU_HIDDEN_CLAMP_MAX)

    def _compute_hidden_stats(self, X_train):
        with torch.no_grad():
            pre_act = self._ph_given_v(X_train)
        return {"h_mean":     F.relu(pre_act).clamp(max=RELU_HIDDEN_CLAMP_MAX).mean().item(),
                "h_sparsity": (pre_act < 0).float().mean().item()}


class NBSigmoidRBM(SigmoidHiddenMonitor, NB_RBM):
    """
    NB-Sigmoid RBM: NB visible units with sigmoid hidden units.

    Hidden units:
      mean_j = sigmoid(b_j + sum_i W_ij * v_i)      h_j ∈ (0,1)
      h_j   ~ Bernoulli(mean_j)                       sampled binary

    Sigmoid is bounded → no dead-unit problem → PCD safe.
    """

    def _ph_given_v(self, V):
        return torch.sigmoid(V @ self.W + self.b)

    def _sample_hidden(self, mean):
        return torch.bernoulli(mean)

    def _sample_bernoulli(self, prob):
        return self._sample_hidden(prob)

    @torch.no_grad()
    def reconstruct(self, V):
        H = self._ph_given_v(V)
        return self._mu(H)


class NBSoftmaxRBM(SoftmaxHiddenMonitor, NB_RBM):
    """
    NB-Softmax RBM: NB visible units with softmax hidden units.

    Hidden units:
      p_j      = softmax_j(b + V @ W)                Σ_j p_j = 1
      h       ~ one-hot(multinomial(p))               exactly one unit active

    Softmax is bounded → PCD safe. The competition between units
    implements a mixture model: each sample is assigned to one
    archetypal community state.
    """

    def _ph_given_v(self, V):
        return torch.softmax(V @ self.W + self.b, dim=1)

    def _sample_hidden(self, mean):
        idx = torch.multinomial(mean, 1).squeeze(1)
        return F.one_hot(idx, num_classes=self.L).float()

    def _sample_bernoulli(self, prob):
        return self._sample_hidden(prob)

    @torch.no_grad()
    def reconstruct(self, V):
        H = self._ph_given_v(V)
        return self._mu(H)
