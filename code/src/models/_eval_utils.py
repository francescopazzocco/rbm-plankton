"""
_eval_utils.py - Shared loss functions and Gibbs imputation for NaN evaluation.

Centralises the NB / ZINB / Bernoulli log-likelihood formulas and clamped-Gibbs
scoring loop that were duplicated across split_comparison.py and nan_test_eval.py.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


# -- Log-likelihood functions --------------------------------------------------


def nb_log_prob(v: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                eps: float = EPS) -> torch.Tensor:
    """NB log-likelihood per (sample, taxon)."""
    return (torch.lgamma(v + theta)
            - torch.lgamma(theta)
            - torch.lgamma(v + 1)
            + theta * torch.log(theta / (theta + mu + eps))
            + v * torch.log(mu / (theta + mu + eps)))


def bern_log_prob(v: torch.Tensor, p: torch.Tensor,
                  eps: float = EPS) -> torch.Tensor:
    """Bernoulli log-likelihood per (sample, taxon)."""
    return v * torch.log(p + eps) + (1 - v) * torch.log(1 - p + eps)


def zinb_log_prob(v: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor,
                  pi: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """ZINB log-likelihood per (sample, taxon)."""
    log_nb_zero = theta * torch.log(theta / (theta + mu + eps))
    log_nb_full = (torch.lgamma(v + theta)
                   - torch.lgamma(theta)
                   - torch.lgamma(v + 1)
                   + theta * torch.log(theta / (theta + mu + eps))
                   + v * torch.log(mu / (theta + mu + eps)))
    log_prob_pos  = torch.log(1 - pi + eps) + log_nb_full
    log_prob_zero = torch.logaddexp(torch.log(pi + eps),
                                    torch.log(1 - pi + eps) + log_nb_zero)
    return torch.where(v > 0, log_prob_pos, log_prob_zero)


# -- Clamped Gibbs scoring loop -----------------------------------------------


@torch.no_grad()
def score_row_gibbs(
    rbm,
    v_raw: np.ndarray,
    device: torch.device,
    *,
    n_samples: int = 100,
    impute_base: int = 5,
    impute_per_nan: int = 3,
    sample_hidden: Callable,
    sample_visible: Callable,
    compute_loss: Callable,
) -> float:
    """Clamped Gibbs imputation + conditional scoring for a single row.

    Parameters
    ----------
    rbm : RBM instance
    v_raw : 1-D array of observed values with NaN for missing taxa
    device : torch.device
    n_samples : multiple stochastic samples for scoring
    impute_base : minimum Gibbs steps
    impute_per_nan : extra Gibbs steps per missing taxon
    sample_hidden : rbm._sample_bernoulli or equivalent
    sample_visible : rbm._mu (NB) or rbm._pv_given_h (Bernoulli) or equiv
    compute_loss : callable(rbm, H, obs_mask, v_obs) -> scalar
    """
    obs = ~np.isnan(v_raw)
    if obs.sum() == 0:
        return float("nan")

    n_steps = impute_base + impute_per_nan * int((~obs).sum())
    obs_t   = torch.tensor(obs, device=device)
    v_raw_t = torch.tensor(v_raw.astype(np.float32), device=device)

    v_curr_t = torch.tensor(np.where(obs, v_raw, 0.0).astype(np.float32),
                            device=device)
    for _ in range(n_steps):
        ph = rbm._ph_given_v(v_curr_t.unsqueeze(0))
        h  = sample_hidden(ph)
        v  = sample_visible(rbm, h).squeeze(0)
        v_curr_t = torch.where(obs_t, v_raw_t, v)

    v_inp_t = v_curr_t.unsqueeze(0).expand(n_samples, -1)
    v_obs_t = v_raw_t[obs_t].unsqueeze(0).expand(n_samples, -1)
    ph = rbm._ph_given_v(v_inp_t)
    H  = sample_hidden(ph)
    return compute_loss(rbm, H, obs_t, v_obs_t)


# -- Per-family scoring closures -----------------------------------------------


def loss_nb(rbm, H: torch.Tensor, obs_t: torch.Tensor,
            v_obs_t: torch.Tensor) -> float:
    mu = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    ll = nb_log_prob(v_obs_t, mu, theta)
    return -ll.mean(dim=1).mean().item()


def loss_zinb(rbm, H: torch.Tensor, obs_t: torch.Tensor,
              v_obs_t: torch.Tensor) -> float:
    mu = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    pi = rbm._pi()[obs_t]
    ll = zinb_log_prob(v_obs_t, mu, theta, pi)
    return -ll.mean(dim=1).mean().item()


def loss_bern(rbm, H: torch.Tensor, obs_t: torch.Tensor,
              v_obs_t: torch.Tensor) -> float:
    pv = rbm._pv_given_h(H)[:, obs_t].clamp(1e-7, 1 - 1e-7)
    loss = F.binary_cross_entropy(pv, v_obs_t, reduction="none")
    return loss.mean(dim=1).mean().item()


def sample_nb(rbm, h: torch.Tensor) -> torch.Tensor:
    return rbm._sample_nb(rbm._mu(h))


def sample_zinb(rbm, h: torch.Tensor) -> torch.Tensor:
    return rbm._sample_zinb(rbm._mu(h))


def sample_bern(rbm, h: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(rbm._pv_given_h(h))
