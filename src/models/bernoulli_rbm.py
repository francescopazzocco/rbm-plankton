"""
bernoulli_rbm.py - Bernoulli-Bernoulli RBM
==========================================
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .base_rbm import BaseRBM


class BernoulliRBM(BaseRBM):
    """Bernoulli-Bernoulli RBM trained by CD-k with RMSprop."""

    def __init__(self, n_visible, n_hidden, device=torch.device("cpu")):
        super().__init__(n_visible, n_hidden, device, scale_init=True)

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
    def hidden_probs(self, V):
        return self._ph_given_v(V)

    @torch.no_grad()
    def pll(self, V):
        """
        Negative pseudo-log-likelihood: -mean_{n,i} log p(v_i | v_{-i}).
        Tractable proxy for NLL; lower is better.

        Derivation: for each unit i, ΔF_i = F(v_i=1) - F(v_i=0) given v_{-i}.
          p(v_i=1 | v_{-i}) = σ(-ΔF_i)
          NPLL = BCE(σ(-ΔF_i), v_i) averaged over all units and samples.
        """
        s      = V @ self.W + self.b                                        # (N, L)
        s_base = s.unsqueeze(1) - V.unsqueeze(2) * self.W.unsqueeze(0)     # (N, D, L)
        delta_F = -self.a - (
            F.softplus(s_base + self.W.unsqueeze(0)) - F.softplus(s_base)
        ).sum(2)                                                             # (N, D)
        return F.binary_cross_entropy(torch.sigmoid(-delta_F), V).item()

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

        history = {"train_mse": [], "val_mse": [],
                   "train_pll": [], "val_pll": [], "epoch": []}

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
                train_pll = self.pll(X_train)
                val_pll   = self.pll(X_val) if X_val is not None else None
                history["epoch"].append(epoch)
                history["train_mse"].append(train_mse)
                history["val_mse"].append(val_mse)
                history["train_pll"].append(train_pll)
                history["val_pll"].append(val_pll)
                if verbose:
                    stats = {"pll": f"{train_pll:.4f}",
                             "batch": batch_size}
                    if val_pll is not None:
                        stats["val_pll"] = f"{val_pll:.4f}"
                    pbar.set_postfix(stats)

        return history

    def numpy_params(self):
        return (self.W.cpu().float().numpy(),
                self.a.cpu().float().numpy(),
                self.b.cpu().float().numpy())
