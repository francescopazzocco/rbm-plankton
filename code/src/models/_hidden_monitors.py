"""
_hidden_monitors.py - Mixins for hidden-unit monitoring in train()
=================================================================
Each mixin provides the four hook methods used by train() so that
the same monitoring logic is defined once and shared across
visible-side variants (NB / ZINB).
"""

import torch
from ._constants import BERNOULLI_SAT_LO, BERNOULLI_SAT_HI, SOFTMAX_ENTROPY_EPS


class BernoulliHiddenMonitor:
    """Mixin: sat_lo / sat_hi / sat_mid for Bernoulli hidden units."""

    def _hidden_stats_keys(self):
        return ["sat_lo", "sat_hi", "sat_mid"]

    def _hidden_stats_init(self):
        return {k: [] for k in self._hidden_stats_keys()}

    def _compute_hidden_stats(self, X_train):
        with torch.no_grad():
            ph = self._ph_given_v(X_train)
        sat_lo  = (ph < BERNOULLI_SAT_LO).float().mean().item()
        sat_hi  = (ph > BERNOULLI_SAT_HI).float().mean().item()
        sat_mid = 1.0 - sat_lo - sat_hi
        return {"sat_lo": sat_lo, "sat_hi": sat_hi, "sat_mid": sat_mid}

    def _hidden_stats_display(self, hid_stats):
        return {"sat_mid": f"{hid_stats['sat_mid']:.0%}"}


class ReLUHiddenMonitor:
    """Mixin: h_mean / h_sparsity for Rectified Gaussian hidden units."""

    def _hidden_stats_keys(self):
        return ["h_mean", "h_sparsity"]

    def _hidden_stats_init(self):
        return {k: [] for k in self._hidden_stats_keys()}

    def _compute_hidden_stats(self, X_train):
        with torch.no_grad():
            h = self._ph_given_v(X_train)
        return {"h_mean": h.mean().item(),
                "h_sparsity": (h == 0).float().mean().item()}

    def _hidden_stats_display(self, hid_stats):
        return {"h_sparsity": f"{hid_stats['h_sparsity']:.0%}",
                "h_mean": f"{hid_stats['h_mean']:.3f}"}


class SigmoidHiddenMonitor:
    """Mixin: h_mean for sigmoid hidden units (target 0.2–0.8)."""

    def _hidden_stats_keys(self):
        return ["h_mean"]

    def _hidden_stats_init(self):
        return {k: [] for k in self._hidden_stats_keys()}

    def _compute_hidden_stats(self, X_train):
        with torch.no_grad():
            h = self._ph_given_v(X_train)
        return {"h_mean": h.mean().item()}

    def _hidden_stats_display(self, hid_stats):
        return {"h_mean": f"{hid_stats['h_mean']:.3f}"}


class SoftmaxHiddenMonitor:
    """Mixin: h_entropy for softmax hidden units (simplex)."""

    def _hidden_stats_keys(self):
        return ["h_entropy"]

    def _hidden_stats_init(self):
        return {k: [] for k in self._hidden_stats_keys()}

    def _compute_hidden_stats(self, X_train):
        with torch.no_grad():
            h = self._ph_given_v(X_train)
        entropy = (-h * torch.log(h.clamp(min=SOFTMAX_ENTROPY_EPS))).sum(1).mean().item()
        return {"h_entropy": entropy}

    def _hidden_stats_display(self, hid_stats):
        return {"H": f"{hid_stats['h_entropy']:.3f}"}
