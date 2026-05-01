"""
base_rbm.py - Base class for RBM models
========================================
Shared interface and initialization logic.
"""

import math
import torch
import torch.nn.functional as F


class BaseRBM:
    """Common RBM interface: init, hidden_probs, reconstruct, numpy_params."""

    def __init__(self, n_visible, n_hidden, device, scale_init=True):
        self.D      = n_visible
        self.L      = n_hidden
        self.device = device

        if scale_init:
            scale = math.sqrt(4.0 / (n_visible + n_hidden))
            self.W = torch.randn(n_visible, n_hidden, device=device) * scale
            self.b = torch.zeros(n_hidden,  device=device)
            self.a = torch.zeros(n_visible, device=device)
        else:
            self.W = None
            self.b = None
            self.a = None

    @torch.no_grad()
    def hidden_probs(self, V):
        raise NotImplementedError

    @torch.no_grad()
    def reconstruct(self, V):
        raise NotImplementedError

    @torch.no_grad()
    def reconstruction_mse(self, V):
        return F.mse_loss(self.reconstruct(V), V).item()

    def numpy_params(self):
        raise NotImplementedError
