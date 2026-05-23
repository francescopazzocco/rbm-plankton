"""
nan_test_eval.py - NaN imputation inference with validated RBMs (L=6).

For each row with missing taxa: Gibbs-impute missing values → score NLL
on observed positions only.

Missingness patterns (after nonzero filter):
  p3:   3 NaN taxa  (104 rows, 80 observed)
  p31: 31 NaN taxa   (43 rows, 52 observed)
  p54: 54 NaN taxa   (13 rows, 29 observed)

Metrics:
  NB / ZINB families (all hidden types): NLL on raw-count observed taxa
  Bernoulli families (median / zero):    BCE on binarised observed taxa

Outputs (all in diagnostic_outputs/nan_eval_extended/):
  nan_eval_rows.csv      per-row NLL, date, missingness pattern
  nan_eval_summary.csv   per-(family, pattern) mean ± std
  nan_eval_bars.png      grouped bar chart — all families
  nan_eval_timeseries.png  p31 NLL time series — all families
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from models.io import best_seed_dir, METRIC_COL, DATA_PATH
from models.utils import load_weights, get_device
from models.nb_rbm import NB_RBM, NBSigmoidRBM, NBSoftmaxRBM
from models.zinb_rbm import ZINB_RBM, ZINBSigmoidRBM, ZINBSoftmaxRBM
from models.bernoulli_rbm import BernoulliRBM


# -- Config ------------------------------------------------------------------

@dataclass
class EvalConfig:
    results_dir: Path   = Path(__file__).parent.parent.parent / "trained_models"
    out_root:    Path   = field(default_factory=lambda:
                            Path(__file__).parent.parent.parent / "diagnostic_outputs" / "nan_eval_extended")
    n_samples:   int    = 100
    impute_base: int    = 5
    impute_per_nan: int = 3
    count_scale: float  = 1000.0
    l: int              = 6


# -- Data --------------------------------------------------------------------

def load_nan_rows() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df = df[df[taxa].fillna(0).sum(axis=1) > 0]
    df = df[df[taxa].isna().any(axis=1)].copy().reset_index(drop=True)
    return df, taxa


def prepare_count(base: pd.DataFrame, taxa: list[str], config: EvalConfig) -> pd.DataFrame:
    out = base.copy()
    out[taxa] = out[taxa] * config.count_scale
    return out


def prepare_bernoulli(base: pd.DataFrame, taxa: list[str], thresholds: np.ndarray) -> pd.DataFrame:
    X = base[taxa].values.astype(np.float32)
    X_bin = (X > thresholds).astype(np.float32)
    X_bin[np.isnan(X)] = np.nan
    out = base.copy()
    out[taxa] = X_bin
    return out


# -- Model loading -----------------------------------------------------------

def _npz_to_tensors(seed_dir: Path, keys: list[str], device) -> dict[str, Any]:
    npz = load_weights(seed_dir / "weights.npz")
    return {k: torch.tensor(npz[k], dtype=torch.float32, device=device) for k in keys}


def load_nb_family(seed_dir: Path, device, cls: Type = NB_RBM) -> NB_RBM:
    p = _npz_to_tensors(seed_dir, ["W", "a", "b", "log_theta"], device)
    rbm = cls(*p["W"].shape, device=device)
    rbm.W, rbm.a, rbm.b, rbm.log_theta = p["W"], p["a"], p["b"], p["log_theta"]
    return rbm


def load_zinb_family(seed_dir: Path, device, cls: Type = ZINB_RBM) -> ZINB_RBM:
    p = _npz_to_tensors(seed_dir, ["W", "a", "b", "log_theta", "logit_pi"], device)
    rbm = cls(*p["W"].shape, device=device)
    rbm.W, rbm.a, rbm.b = p["W"], p["a"], p["b"]
    rbm.log_theta = p["log_theta"]
    rbm.logit_pi = p["logit_pi"]
    return rbm


def load_bernoulli(seed_dir: Path, device) -> tuple[BernoulliRBM, np.ndarray]:
    npz_raw = load_weights(seed_dir / "weights.npz")
    p = {k: torch.tensor(npz_raw[k], dtype=torch.float32, device=device)
         for k in ["W", "a", "b"]}
    rbm = BernoulliRBM(*p["W"].shape, device=device)
    rbm.W, rbm.a, rbm.b = p["W"], p["a"], p["b"]
    return rbm, npz_raw["thresholds"]


def read_val(seed_dir: Path, col: str) -> float:
    return pd.read_csv(seed_dir / "rbm_training_curves.csv")[col].dropna().iloc[-1]


# -- Core: Gibbs imputation + conditional scoring ---------------------------

def _bernoulli_sample(prob: torch.Tensor) -> torch.Tensor:
    return (torch.rand_like(prob) < prob).float()


def score_row(
    rbm,
    v_raw: np.ndarray,
    config: EvalConfig,
    device,
    sample_hidden: Callable,
    sample_visible: Callable,
    compute_loss: Callable,
) -> float:
    obs = ~np.isnan(v_raw)
    if obs.sum() == 0:
        return float("nan")

    n_steps = config.impute_base + config.impute_per_nan * int((~obs).sum())
    obs_t   = torch.tensor(obs, device=device)
    v_raw_t = torch.tensor(v_raw.astype(np.float32), device=device)

    v_curr_t = torch.tensor(np.where(obs, v_raw, 0.0).astype(np.float32), device=device)
    for _ in range(n_steps):
        ph = rbm._ph_given_v(v_curr_t.unsqueeze(0))
        h  = sample_hidden(ph)
        v  = sample_visible(rbm, h).squeeze(0)
        v_curr_t = torch.where(obs_t, v_raw_t, v)

    v_inp_t = v_curr_t.unsqueeze(0).expand(config.n_samples, -1)
    v_obs_t = v_raw_t[obs_t].unsqueeze(0).expand(config.n_samples, -1)
    ph = rbm._ph_given_v(v_inp_t)
    H  = sample_hidden(ph)
    return compute_loss(rbm, H, obs_t, v_obs_t)


# -- NB loss -----------------------------------------------------------------

def _loss_nb(rbm: NB_RBM, H: torch.Tensor, obs_t: torch.Tensor, v_obs_t: torch.Tensor) -> float:
    mu    = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    eps   = 1e-8
    log_nb = (torch.lgamma(v_obs_t + theta)
              - torch.lgamma(theta)
              - torch.lgamma(v_obs_t + 1)
              + theta * torch.log(theta / (theta + mu + eps))
              + v_obs_t * torch.log(mu   / (theta + mu + eps)))
    return -log_nb.mean(dim=1).mean().item()


def _sample_visible_nb(rbm: NB_RBM, h: torch.Tensor) -> torch.Tensor:
    return rbm._sample_nb(rbm._mu(h))


def score_nb_family(rbm: NB_RBM, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row(rbm, v_raw, config, device,
                     rbm._sample_bernoulli, _sample_visible_nb, _loss_nb)


# -- ZINB loss ---------------------------------------------------------------

def _loss_zinb(rbm: ZINB_RBM, H: torch.Tensor, obs_t: torch.Tensor, v_obs_t: torch.Tensor) -> float:
    mu    = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    pi    = rbm._pi()[obs_t]
    eps   = 1e-8

    log_nb_zero = theta * torch.log(theta / (theta + mu + eps))
    log_nb_full = (torch.lgamma(v_obs_t + theta)
                   - torch.lgamma(theta)
                   - torch.lgamma(v_obs_t + 1)
                   + theta * torch.log(theta / (theta + mu + eps))
                   + v_obs_t * torch.log(mu   / (theta + mu + eps)))

    log_prob_pos  = torch.log(1 - pi + eps) + log_nb_full
    log_prob_zero = torch.logaddexp(torch.log(pi + eps),
                                    torch.log(1 - pi + eps) + log_nb_zero)
    log_prob = torch.where(v_obs_t > 0, log_prob_pos, log_prob_zero)
    return -log_prob.mean(dim=1).mean().item()


def _sample_visible_zinb(rbm: ZINB_RBM, h: torch.Tensor) -> torch.Tensor:
    return rbm._sample_zinb(rbm._mu(h))


def score_zinb_family(rbm: ZINB_RBM, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row(rbm, v_raw, config, device,
                     rbm._sample_bernoulli, _sample_visible_zinb, _loss_zinb)


# -- Bernoulli loss ----------------------------------------------------------

def _loss_bern(rbm: BernoulliRBM, H: torch.Tensor, obs_t: torch.Tensor, v_obs_t: torch.Tensor) -> float:
    pv = rbm._pv_given_h(H)[:, obs_t].clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(pv, v_obs_t, reduction="none").mean(dim=1).mean().item()


def _sample_visible_bern(rbm: BernoulliRBM, h: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(rbm._pv_given_h(h))


def score_bern(rbm: BernoulliRBM, v_raw: np.ndarray, config: EvalConfig, device) -> float:
    return score_row(rbm, v_raw, config, device,
                     _bernoulli_sample, _sample_visible_bern, _loss_bern)


# -- Evaluation loop ---------------------------------------------------------

def evaluate(rbm, rows_df: pd.DataFrame, taxa: list[str],
             config: EvalConfig, device, score_fn: Callable) -> pd.DataFrame:
    records = []
    for _, row in rows_df.iterrows():
        v = row[taxa].values.astype(np.float32)
        n_miss = int(np.isnan(v).sum())
        records.append({
            "date":   row["date"],
            "n_obs":  len(v) - n_miss,
            "n_miss": n_miss,
            "nll":    score_fn(rbm, v, config, device),
        })
    return pd.DataFrame(records)


# -- Aggregation -------------------------------------------------------------

PATTERN_MAP = {3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["family", "pattern", "n_miss", "n_obs"])
            .agg(n_rows=("nll", "count"), nll_mean=("nll", "mean"), nll_std=("nll", "std"))
            .reset_index().sort_values(["family", "n_miss"]))


# -- Figures -----------------------------------------------------------------

PATTERN_LABELS = {
    "p3_3miss":   "3 missing\n(n=104)",
    "p31_31miss": "31 missing\n(n=43)",
    "p54_54miss": "54 missing\n(n=13)",
}
PATTERNS = ["p3_3miss", "p31_31miss", "p54_54miss"]

FAMILY_CFG = {
    "nb":               {"color": "#08519c", "label": "NB-chron (L=6)"},
    "nb_shuffled":      {"color": "#5e3c99", "label": "NB-shuffled (L=8)"},
    "nb_sigmoid":       {"color": "#3182bd", "label": "NB-Sigmoid (L=7)"},
    "nb_softmax":       {"color": "#9ecae1", "label": "NB-Softmax (L=7)"},
    "zinb":             {"color": "#006d2c", "label": "ZINB (L=8)"},
    "zinb_sigmoid":     {"color": "#31a354", "label": "ZINB-Sigmoid (L=7)"},
    "zinb_softmax":     {"color": "#a1d99b", "label": "ZINB-Softmax (L=6)"},
    "bernoulli_median": {"color": "#e6550d", "label": "Bernoulli-median (L=6)"},
    "bernoulli_zero":   {"color": "#fdae6b", "label": "Bernoulli-zero (L=6)"},
}


def plot_bars(summary: pd.DataFrame, out: Path):
    families = [f for f in FAMILY_CFG if f in summary["family"].unique()]
    n_fam = len(families)
    w = 0.75 / n_fam
    x = np.arange(len(PATTERNS))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, fam in enumerate(families):
        cfg = FAMILY_CFG[fam]
        sub = summary[summary["family"] == fam].set_index("pattern")
        means = [sub.loc[p, "nll_mean"] if p in sub.index else np.nan for p in PATTERNS]
        stds  = [sub.loc[p, "nll_std"]  if p in sub.index else 0.0    for p in PATTERNS]
        offset = (i - (n_fam - 1) / 2) * w
        ax.bar(x + offset, means, w, yerr=stds, capsize=3,
               color=cfg["color"], alpha=0.85, label=cfg["label"],
               error_kw={"linewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS[p] for p in PATTERNS])
    ax.set_xlabel("Missingness pattern")
    ax.set_ylabel("NLL on observed taxa\n(NB/ZINB: count NLL  |  Bernoulli: BCE)")
    ax.set_title("NaN test set evaluation — all model families (optimal L per family)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out: Path):
    p31 = df[df["n_miss"] == 31].copy()
    p31["date"] = pd.to_datetime(p31["date"])
    p31 = p31.sort_values("date")
    families = [f for f in FAMILY_CFG if f in p31["family"].unique()]

    fig, ax = plt.subplots(figsize=(12, 4))
    for fam in families:
        cfg = FAMILY_CFG[fam]
        sub = p31[p31["family"] == fam]
        ax.plot(sub["date"], sub["nll"], "o-", color=cfg["color"],
                markersize=3, linewidth=1, label=cfg["label"], alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("NLL / BCE (52 observed taxa)")
    ax.set_title("p31 pattern — per-day test NLL, all model families (optimal L per family)")
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -- Main --------------------------------------------------------------------

# (family_name, dir_name, metric_key, model_cls, is_zinb)
# Optimal L per family: nb_chron=6, nb_shuffled=8, nb_sigmoid=7, nb_softmax=7,
#                       zinb=8, zinb_sigmoid=7, zinb_softmax=6, bernoulli*=6
# is_zinb=None → Bernoulli family
_SPECS = [
    ("nb",               "nb_L6",                     "nb",               NB_RBM,         False),
    ("nb_shuffled",      "nb_L8_shuffled",             "nb",               NB_RBM,         False),
    ("nb_sigmoid",       "nb_sigmoid_L7_shuffled",    "nb_sigmoid",       NBSigmoidRBM,   False),
    ("nb_softmax",       "nb_softmax_L7_shuffled",    "nb_softmax",       NBSoftmaxRBM,   False),
    ("zinb",             "zinb_L8_shuffled",           "zinb",             ZINB_RBM,       True),
    ("zinb_sigmoid",     "zinb_sigmoid_L7_shuffled",  "zinb_sigmoid",     ZINBSigmoidRBM, True),
    ("zinb_softmax",     "zinb_softmax_L6_shuffled",  "zinb_softmax",     ZINBSoftmaxRBM, True),
    ("bernoulli_median", "bernoulli_median_L6",        "bernoulli_median", None,           None),
    ("bernoulli_zero",   "bernoulli_zero_L6",          "bernoulli_zero",   None,           None),
]


def main():
    config = EvalConfig()
    config.out_root.mkdir(parents=True, exist_ok=True)
    device = get_device()

    nan_df, taxa = load_nan_rows()
    count_rows = prepare_count(nan_df, taxa, config)

    all_dfs = []
    for fam_name, dir_name, metric_key, cls, is_zinb in _SPECS:
        fam_dir  = config.results_dir / dir_name
        if not fam_dir.exists():
            print(f"\n[SKIP] {dir_name} not found")
            continue

        seed_dir = best_seed_dir(fam_dir, METRIC_COL[metric_key])
        if seed_dir is None:
            print(f"\n[SKIP] no valid seed in {dir_name}")
            continue

        val_col = METRIC_COL[metric_key]
        print(f"\n=== {fam_name} ===")
        print(f"  dir: {dir_name}  seed: {seed_dir.name}  ({val_col} = {read_val(seed_dir, val_col):.4f})")

        if is_zinb is None:
            rbm, thresholds = load_bernoulli(seed_dir, device)
            rows_df  = prepare_bernoulli(nan_df, taxa, thresholds)
            score_fn = score_bern
        elif is_zinb:
            rbm      = load_zinb_family(seed_dir, device, cls=cls)
            rows_df  = count_rows
            score_fn = score_zinb_family
        else:
            rbm      = load_nb_family(seed_dir, device, cls=cls)
            rows_df  = count_rows
            score_fn = score_nb_family

        df_fam = evaluate(rbm, rows_df, taxa, config, device, score_fn)
        df_fam["family"] = fam_name
        all_dfs.append(df_fam)
        print(f"  evaluated {len(df_fam)} rows")

    df = pd.concat(all_dfs, ignore_index=True)
    df["pattern"] = df["n_miss"].map(PATTERN_MAP)

    rows_out   = config.out_root / "nan_eval_rows.csv"
    summary    = summarise(df)
    summary_out = config.out_root / "nan_eval_summary.csv"
    df.to_csv(rows_out, index=False)
    summary.to_csv(summary_out, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- NaN test set evaluation summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved:    {rows_out}")
    print(f"Summary saved: {summary_out}")

    plot_bars(summary, config.out_root / "nan_eval_bars.png")
    plot_timeseries(df, config.out_root / "nan_eval_timeseries.png")
    print(f"Figures saved: {config.out_root}/nan_eval_bars.png  &  nan_eval_timeseries.png")


if __name__ == "__main__":
    main()
