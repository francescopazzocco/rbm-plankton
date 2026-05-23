"""
zinb_rbm.py - Zero-Inflated Negative Binomial RBM
================================================
ZINB visible units with Bernoulli hidden units.
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


class ZINB_RBM(BernoulliHiddenMonitor, BaseRBM):
    """
    ZINB-Bernoulli RBM trained by CD-k with RMSprop.

    Visible units: Zero-Inflated Negative Binomial
      p(v_i | h) = pi_i * delta_0(v) + (1 - pi_i) * NB(v_i; mu_i, theta_i)
      mu_i = exp(a_i + sum_j W_ij * h_j)           <- exp ensures positivity
      theta_i = exp(log_theta_i)                    <- learned per-taxon dispersion
      pi_i = sigmoid(logit_pi_i)                    <- learned per-taxon inflation prob

    Hidden units: Bernoulli
      p(h_j=1 | v) = sigmoid(b_j + sum_i W_ij * v_i)

    CD-k gradient derivation:
      d log ZINB(v_i|h) / d eta_i
        = theta_i * (v_i - mu_i) / (mu_i + theta_i)        if v > 0
        = -(1-pi_i) * theta_i * mu_i * f_i / ((mu_i+theta_i) * (pi_i + (1-pi_i)*f_i))
                                                            if v = 0
      where f_i = (theta_i / (theta_i + mu_i))^theta_i
            eta_i = a_i + W_i * h

      pi and theta are updated via autograd on positive-phase NLL.
    """

    def __init__(self, n_visible, n_hidden,
                 device=torch.device("cpu"),
                 theta_init_log=0.0,
                 logit_pi_init=-2.0):
        super().__init__(n_visible, n_hidden, device, scale_init=True)
        self.log_theta = torch.full((n_visible,), theta_init_log,
                                    device=device, requires_grad=False)
        self.logit_pi = torch.full((n_visible,), logit_pi_init,
                                   device=device, requires_grad=False)

    # --- internal helpers ---

    def _eta(self, H):
        """Linear predictor: n = a + H @ W.T  ->  shape (batch, D)"""
        return self.a.unsqueeze(0) + H @ self.W.t()

    def _mu(self, H):
        """NB mean: mu_i = exp(n_i), clamped to prevent float32 overflow."""
        return torch.exp(self._eta(H).clamp(max=ETA_CLAMP_MAX))

    def _pi(self):
        """Zero-inflation probability per taxon."""
        return torch.sigmoid(self.logit_pi)

    def _ph_given_v(self, V):
        """P(H=1|V) = sigmoid(b + V @ W),  shape (batch, L)"""
        return torch.sigmoid(V @ self.W + self.b)

    @staticmethod
    def _sample_bernoulli(prob):
        return (torch.rand_like(prob) < prob).float()

    def _sample_zinb(self, mu):
        """
        Sample from ZINB(mu, theta, pi):
          z ~ Bernoulli(pi)   -> if z=1, v=0
          otherwise v ~ NB(mu, theta) via Gamma-Poisson mixture
        """
        pi = self._pi().detach()
        theta = self.log_theta.detach().exp().clamp(min=THETA_CLAMP_MIN)

        z = (torch.rand_like(mu) < pi.unsqueeze(0).expand_as(mu)).float()

        concentration = theta.unsqueeze(0).expand_as(mu)
        rate = theta.unsqueeze(0) / mu.clamp(min=LOG_PROB_EPS)
        g = torch.distributions.Gamma(concentration, rate).sample()
        nb_samples = torch.poisson(g).float()

        return (1 - z) * nb_samples

    def _zinb_log_prob(self, V, mu):
        """
        ZINB log-likelihood:
          log p(v) = log(pi + (1-pi) * NB(0; mu, theta))    if v=0
                  = log(1-pi) + log NB(v; mu, theta)         if v>0
        Shape: (batch, D) -> scalar (mean over batch and taxa)
        """
        pi = self._pi()
        theta = self.log_theta.exp().clamp(min=THETA_CLAMP_MIN)
        eps = LOG_PROB_EPS

        log_nb_zero = theta * torch.log(theta / (theta + mu + eps))

        log_prob = torch.zeros_like(V)

        mask_zero = V == 0
        mask_pos = V > 0

        if mask_zero.any():
            log_pi = torch.log(pi + eps)
            log_1minuspi_f = torch.log(1 - pi + eps) + log_nb_zero
            log_prob[mask_zero] = torch.logaddexp(log_pi, log_1minuspi_f)[mask_zero]

        if mask_pos.any():
            log_nb = (torch.lgamma(V + theta)
                      - torch.lgamma(theta)
                      - torch.lgamma(V + 1)
                      + theta * torch.log(theta / (theta + mu + eps))
                      + V     * torch.log(mu    / (theta + mu + eps)))
            log_prob[mask_pos] = (torch.log(1 - pi + eps) + log_nb)[mask_pos]

        return log_prob.mean()

    def _zinb_residual(self, V, mu, pi=None):
        """
        Weighted residual r_i = d log ZINB(v_i|h) / d eta_i.
        Used in CD gradients for W and a.
        Shape: (batch, D)
        """
        if pi is None:
            pi = self._pi().detach()
        theta = self.log_theta.detach().exp().clamp(min=THETA_CLAMP_MIN)
        eps = LOG_PROB_EPS

        nb_res = theta * (V - mu) / (mu + theta + eps)
        log_f = theta * torch.log(theta / (theta + mu + eps))
        f = torch.exp(log_f)
        p0 = pi + (1 - pi) * f
        zero_res = -(1 - pi) * theta * mu * f / ((mu + theta + eps) * (p0 + eps))

        return torch.where(V > 0, nb_res, zero_res)

    # --- public interface ---

    @torch.no_grad()
    def reconstruct(self, V):
        """V -> h sample -> ZINB conditional mean (1-pi)*mu"""
        ph = self._ph_given_v(V)
        H  = self._sample_bernoulli(ph)
        mu = self._mu(H)
        pi = self._pi()
        return (1 - pi.unsqueeze(0)) * mu

    @torch.no_grad()
    def hidden_probs(self, V):
        return self._ph_given_v(V)

    def nll(self, V):
        """Negative log-likelihood on V (positive phase only, no CD)."""
        with torch.no_grad():
            ph = self._ph_given_v(V)
            H  = self._sample_bernoulli(ph)
        mu = self._mu(H)
        return -self._zinb_log_prob(V, mu).item()

    def train(self, X_train, X_val=None,
              epochs=500, lr=0.01, lr_decay=DEFAULT_LR_DECAY,
              cd_steps=1, batch_i=DEFAULT_BATCH_I, batch_f=DEFAULT_BATCH_F, n_batches=DEFAULT_N_BATCHES,
              gamma=DEFAULT_GAMMA, beta=DEFAULT_BETA, epsilon=DEFAULT_RMSPROP_EPS,
              lr_theta=None, lr_pi=None,
              use_pcd=False, n_pcd_chains=500,
              eval_every=10, verbose=True):
        """
        CD-k / PCD-k training for ZINB-Bernoulli RBM.

        Parameters
        ----------
        lr_theta : float | None
            Learning rate for theta (dispersion). If None, uses lr * 0.1.
        lr_pi : float | None
            Learning rate for logit_pi (inflation). If None, uses lr * 0.1.
        use_pcd : bool
            Use Persistent CD instead of standard CD.
        n_pcd_chains : int
            Number of persistent fantasy particles. Must be >= batch_f.
        """
        N           = X_train.shape[0]
        current_lr  = lr
        lr_theta    = lr_theta or lr * LR_PARAM_MULTIPLIER
        lr_pi       = lr_pi or lr * LR_PARAM_MULTIPLIER

        data_mean = X_train.mean(0).clamp(min=LOG_PROB_EPS)
        self.a    = torch.log(data_mean)

        sW = torch.zeros_like(self.W)
        sa = torch.zeros_like(self.a)
        sb = torch.zeros_like(self.b)
        s_theta = torch.zeros_like(self.log_theta.data)
        s_pi    = torch.zeros_like(self.logit_pi.data)

        history = {"train_mse": [], "val_mse": [], "train_nll": [],
                   "val_nll": [], "theta_mean": [], "pi_mean": [], "epoch": []}
        history.update(self._hidden_stats_init())

        if use_pcd:
            pcd_init = torch.randperm(N, device=self.device)[:n_pcd_chains]
            V_pcd = X_train[pcd_init].clone()

        pbar = tqdm(range(1, epochs + 1), desc="Training RBM [ZINB]",
                    unit="epoch")
        for epoch in pbar:
            q          = (epoch - 1) / max(epochs - 1, 1)
            batch_size = int(batch_i + (batch_f - batch_i) * q**2)
            recon_acc  = 0.0

            for _ in range(n_batches):
                idx = torch.randperm(N, device=self.device)[:batch_size]
                V0  = X_train[idx]

                pi_val = self._pi().detach()

                ph0 = self._ph_given_v(V0)
                H0  = self._sample_bernoulli(ph0)
                mu0 = self._mu(H0)
                r0  = self._zinb_residual(V0, mu0, pi=pi_val)

                if use_pcd:
                    sel = torch.randint(0, n_pcd_chains, (batch_size,),
                                        device=self.device)
                    Vk = V_pcd[sel]
                    for _ in range(cd_steps):
                        phk = self._ph_given_v(Vk)
                        Hk  = self._sample_bernoulli(phk)
                        Vk  = self._sample_zinb(self._mu(Hk))
                    V_pcd[sel] = Vk.detach()
                    phk = self._ph_given_v(Vk)
                    Hk  = self._sample_bernoulli(phk)
                else:
                    Hk = H0
                    for _ in range(cd_steps):
                        Vk  = self._sample_zinb(self._mu(Hk))
                        phk = self._ph_given_v(Vk)
                        Hk  = self._sample_bernoulli(phk)

                muk = self._mu(Hk)
                rk  = self._zinb_residual(Vk, muk, pi=pi_val)

                dW = (r0.t() @ ph0  - rk.t() @ phk) / batch_size
                da = (r0 - rk).mean(0)
                db = (ph0 - phk).mean(0)

                recon_acc += F.mse_loss(
                    (1 - pi_val.unsqueeze(0)) * mu0, V0
                ).item()

                sW = beta * sW + (1 - beta) * dW.pow(2)
                sa = beta * sa + (1 - beta) * da.pow(2)
                sb = beta * sb + (1 - beta) * db.pow(2)

                self.W += current_lr * dW / (sW + epsilon).sqrt()
                self.a += current_lr * da / (sa + epsilon).sqrt()
                self.b += current_lr * db / (sb + epsilon).sqrt()

                if gamma > 0:
                    self.W -= gamma * current_lr * self.W.sign()

                # combined theta + pi update via single autograd pass
                self.log_theta.requires_grad_(True)
                self.logit_pi.requires_grad_(True)
                mu0_for_params = self._mu(H0.detach())
                nll = -self._zinb_log_prob(V0, mu0_for_params)
                nll.backward()

                with torch.no_grad():
                    g_theta = self.log_theta.grad.nan_to_num(nan=0.0).clone()
                    g_pi = self.logit_pi.grad.nan_to_num(nan=0.0).clone()
                    s_theta = beta * s_theta + (1 - beta) * g_theta.pow(2)
                    s_pi = beta * s_pi + (1 - beta) * g_pi.pow(2)
                    self.log_theta -= lr_theta * g_theta / (s_theta + epsilon).sqrt()
                    self.logit_pi -= lr_pi * g_pi / (s_pi + epsilon).sqrt()
                    self.log_theta.clamp_(LOG_PARAM_CLAMP_MIN, LOG_PARAM_CLAMP_MAX)
                    self.logit_pi.clamp_(LOG_PARAM_CLAMP_MIN, LOG_PARAM_CLAMP_MAX)
                    self.log_theta.grad.zero_()
                    self.logit_pi.grad.zero_()
                self.log_theta.requires_grad_(False)
                self.logit_pi.requires_grad_(False)

            current_lr *= lr_decay

            if epoch % eval_every == 0 or epoch == 1:
                train_mse   = recon_acc / n_batches
                val_mse     = self.reconstruction_mse(X_val) \
                              if X_val is not None else None
                train_nll   = self.nll(X_train)
                val_nll     = self.nll(X_val) if X_val is not None else None
                theta_mean  = self.log_theta.detach().exp().mean().item()
                pi_mean     = self._pi().detach().mean().item()

                hid_stats = self._compute_hidden_stats(X_train)

                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                history["train_nll"].append(train_nll)
                history["val_nll"].append(val_nll)
                history["theta_mean"].append(theta_mean)
                history["pi_mean"].append(pi_mean)
                for k, v in hid_stats.items():
                    history[k].append(v)

                if verbose:
                    stats = {"nll": f"{train_nll:.2f}",
                             "theta_mean": f"{theta_mean:.3f}",
                             "pi_mean": f"{pi_mean:.3f}",
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
                self.log_theta.detach().cpu().float().numpy(),
                self.logit_pi.detach().cpu().float().numpy())


class ZINB_ReLU_RBM(ReLUHiddenMonitor, ZINB_RBM):
    """
    ZINB-ReLU RBM: ZINB visible units with Rectified Gaussian hidden units.

    Hidden units (replaces Bernoulli):
      mean_j = ReLU(b_j + sum_i W_ij * v_i)
      h_j ~ ReLU(mean_j + N(0,1))   (rectified Gaussian)

    All visible-side logic (ZINB likelihood, theta/pi update, PCD buffer)
    is inherited unchanged from ZINB_RBM.
    """

    def _ph_given_v(self, V):
        return F.relu(V @ self.W + self.b).clamp(max=RELU_HIDDEN_CLAMP_MAX)

    def _sample_hidden(self, mean):
        return F.relu(mean + torch.randn_like(mean)).clamp(max=RELU_HIDDEN_CLAMP_MAX)

    def _sample_bernoulli(self, prob):
        """Overridden: rectified Gaussian sampling in place of Bernoulli."""
        return self._sample_hidden(prob)

    def hidden_probs(self, V):
        return self._ph_given_v(V)


class ZINBSigmoidRBM(SigmoidHiddenMonitor, ZINB_RBM):
    def _ph_given_v(self, V):
        return torch.sigmoid(V @ self.W + self.b)

    def _sample_hidden(self, mean):
        return torch.bernoulli(mean)

    def _sample_bernoulli(self, prob):
        return self._sample_hidden(prob)

    @torch.no_grad()
    def reconstruct(self, V):
        H = self._ph_given_v(V)
        return (1 - self._pi().unsqueeze(0)) * self._mu(H)


class ZINBSoftmaxRBM(SoftmaxHiddenMonitor, ZINB_RBM):
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
        return (1 - self._pi().unsqueeze(0)) * self._mu(H)
