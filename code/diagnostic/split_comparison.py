"""
split_comparison.py - Head-to-head NaN inference test: chronological vs shuffled split.

NaN rows are removed before the train/val split in both strategies (io._base_load),
so all 160 rows are genuinely unseen for every model here.

Models:
  NB-RBM          chrono L=6   vs  shuffled L=8
  Bernoulli-med   chrono L=6   vs  shuffled L=8

Metric: NLL on observed (non-NaN) positions only.
Inference: clamped Gibbs (N_GIBBS steps) — observed positions stay fixed, missing
positions are iteratively inferred from the model before the final scoring step.

Output:
  results/tables/split_comparison.csv
  diagnostic_outputs/03_evaluation/split_comparison.png
"""

import sys
from pathlib import Path

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
from models.nb_rbm import NB_RBM
from models.bernoulli_rbm import BernoulliRBM

RESULTS_DIR = Path(__file__).parent.parent.parent / "trained_models"
OUT_DIR     = Path(__file__).parent.parent.parent / "results" / "tables"
FIG_DIR     = Path(__file__).parent.parent.parent / "diagnostic_outputs" / "03_evaluation"
N_SAMPLES   = 100
N_GIBBS     = 5      # clamped Gibbs steps to infer missing positions before scoring
COUNT_SCALE = 1000

CONFIGS = [
    dict(family="nb",               split="chrono",   L=6),
    dict(family="nb",               split="shuffled", L=8),
    dict(family="bernoulli_median", split="chrono",   L=6),
    dict(family="bernoulli_median", split="shuffled", L=8),
]

PATTERN_LABELS = {
    "p3_3miss":   "3 missing\n(n=104)",
    "p31_31miss": "31 missing\n(n=43)",
    "p54_54miss": "54 missing\n(n=13)",
}
PATTERNS = ["p3_3miss", "p31_31miss", "p54_54miss"]


# -- Directory helpers ---------------------------------------------------------

def run_dir(family: str, L: int, split: str) -> Path:
    suffix = "_shuffled" if split == "shuffled" else ""
    return RESULTS_DIR / f"{family}_L{L}{suffix}"


# -- Model loading -------------------------------------------------------------

def load_nb(seed_dir: Path, device) -> NB_RBM:
    npz = load_weights(seed_dir / "weights.npz")
    W  = torch.tensor(npz["W"],          dtype=torch.float32, device=device)
    a  = torch.tensor(npz["a"],          dtype=torch.float32, device=device)
    b  = torch.tensor(npz["b"],          dtype=torch.float32, device=device)
    lt = torch.tensor(npz["log_theta"],  dtype=torch.float32, device=device)
    rbm = NB_RBM(W.shape[0], W.shape[1], device=device)
    rbm.W, rbm.a, rbm.b, rbm.log_theta = W, a, b, lt
    return rbm


def load_bernoulli(seed_dir: Path, device) -> tuple[BernoulliRBM, np.ndarray]:
    npz = load_weights(seed_dir / "weights.npz")
    W = torch.tensor(npz["W"], dtype=torch.float32, device=device)
    a = torch.tensor(npz["a"], dtype=torch.float32, device=device)
    b = torch.tensor(npz["b"], dtype=torch.float32, device=device)
    rbm = BernoulliRBM(W.shape[0], W.shape[1], device=device)
    rbm.W, rbm.a, rbm.b = W, a, b
    return rbm, npz["thresholds"]


# -- Data loading --------------------------------------------------------------

def nan_rows_nb() -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df   = df[df[taxa].fillna(0).sum(axis=1) > 0].copy()
    rows = df[df[taxa].isna().any(axis=1)].copy().reset_index(drop=True)
    rows[taxa] = rows[taxa] * COUNT_SCALE
    return rows, taxa


def nan_rows_bernoulli(thresholds: np.ndarray) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa = [c for c in df.columns if c != "date"]
    df   = df[df[taxa].fillna(0).sum(axis=1) > 0].copy()
    rows = df[df[taxa].isna().any(axis=1)].copy().reset_index(drop=True)
    X     = rows[taxa].values.astype(np.float32)
    X_bin = (X > thresholds).astype(np.float32)
    X_bin[np.isnan(X)] = np.nan
    rows[taxa] = X_bin
    return rows, taxa


# -- Scoring -------------------------------------------------------------------

@torch.no_grad()
def score_nb_row(rbm: NB_RBM, v_raw: np.ndarray, device) -> float:
    obs = ~np.isnan(v_raw)
    if obs.sum() == 0:
        return float("nan")
    obs_t   = torch.tensor(obs, device=device)
    v_obs_t = torch.tensor(v_raw[obs].astype(np.float32), device=device).unsqueeze(0).expand(N_SAMPLES, -1)
    # Initialise with zeros at missing positions; clamped Gibbs will fill them in
    v_t = torch.tensor(np.where(obs, v_raw, 0.0).astype(np.float32),
                       device=device).unsqueeze(0).expand(N_SAMPLES, -1).clone()
    # Clamped Gibbs: infer missing positions, keep observed clamped
    for _ in range(N_GIBBS):
        ph = rbm._ph_given_v(v_t)
        H  = rbm._sample_bernoulli(ph)
        v_t[:, ~obs_t] = rbm._mu(H)[:, ~obs_t]
    # Score observed positions using hidden state from refined v
    ph  = rbm._ph_given_v(v_t)
    H   = rbm._sample_bernoulli(ph)
    mu  = rbm._mu(H)[:, obs_t]
    theta = rbm.log_theta[obs_t].exp().clamp(min=1e-4)
    eps   = 1e-8
    log_nb = (torch.lgamma(v_obs_t + theta)
              - torch.lgamma(theta)
              - torch.lgamma(v_obs_t + 1)
              + theta * torch.log(theta / (theta + mu + eps))
              + v_obs_t * torch.log(mu    / (theta + mu + eps)))
    return -log_nb.mean(dim=1).mean().item()


@torch.no_grad()
def score_bernoulli_row(rbm: BernoulliRBM, v_bin: np.ndarray, device) -> float:
    obs = ~np.isnan(v_bin)
    if obs.sum() == 0:
        return float("nan")
    obs_t   = torch.tensor(obs, device=device)
    v_obs_t = torch.tensor(v_bin[obs].astype(np.float32), device=device).unsqueeze(0).expand(N_SAMPLES, -1)
    v_t = torch.tensor(np.where(obs, v_bin, 0.0).astype(np.float32),
                       device=device).unsqueeze(0).expand(N_SAMPLES, -1).clone()
    # Clamped Gibbs: infer missing positions, keep observed clamped
    for _ in range(N_GIBBS):
        ph = rbm._ph_given_v(v_t)
        H  = rbm._sample(ph)
        v_t[:, ~obs_t] = rbm._pv_given_h(H)[:, ~obs_t]
    # Score observed positions using hidden state from refined v
    ph  = rbm._ph_given_v(v_t)
    H   = rbm._sample(ph)
    pv  = rbm._pv_given_h(H)[:, obs_t].clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(pv, v_obs_t, reduction="none")
    return bce.mean(dim=1).mean().item()


def evaluate(rows_df, taxa, score_fn, device) -> pd.DataFrame:
    records = []
    for _, row in rows_df.iterrows():
        v   = row[taxa].values.astype(np.float32)
        obs = ~np.isnan(v)
        records.append({
            "date":   row["date"],
            "n_obs":  int(obs.sum()),
            "n_miss": int((~obs).sum()),
            "nll":    score_fn(v, device),
        })
    return pd.DataFrame(records)


# -- Figure --------------------------------------------------------------------

FAMILY_COLORS = {
    "nb":               {"chrono": "#6baed6", "shuffled": "#2171b5"},
    "bernoulli_median": {"chrono": "#fdae6b", "shuffled": "#e6550d"},
}
FAMILY_LABELS = {"nb": "NB-RBM", "bernoulli_median": "Bernoulli-med"}


def plot_comparison(summary: pd.DataFrame, out: Path):
    n_patterns = len(PATTERNS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for ax, family in zip(axes, ["nb", "bernoulli_median"]):
        sub = summary[summary["family"] == family]
        x   = np.arange(n_patterns)
        w   = 0.35
        for i, split in enumerate(["chrono", "shuffled"]):
            s = sub[sub["split"] == split].set_index("pattern")
            means = [s.loc[p, "nll_mean"] if p in s.index else np.nan for p in PATTERNS]
            stds  = [s.loc[p, "nll_std"]  if p in s.index else 0      for p in PATTERNS]
            color = FAMILY_COLORS[family][split]
            L_val = s["L"].iloc[0] if not s.empty else "?"
            label = f"{split}  L={L_val}"
            ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                   color=color, alpha=0.88, label=label,
                   error_kw={"linewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels([PATTERN_LABELS[p] for p in PATTERNS])
        ax.set_xlabel("Missingness pattern")
        ax.set_ylabel("Test NLL (observed taxa only)")
        ax.set_title(FAMILY_LABELS[family])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle("Split strategy comparison — NaN inference test", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure: {out}")


# -- Main ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device   = get_device()
    all_dfs  = []

    for cfg in CONFIGS:
        family, split, L = cfg["family"], cfg["split"], cfg["L"]
        tag = f"{family}  {split}  L={L}"
        print(f"\n=== {tag} ===")

        rdir = run_dir(family, L, split)
        if not rdir.exists():
            print(f"  ! directory not found: {rdir}  — skipping")
            continue

        seed_dir = best_seed_dir(rdir, METRIC_COL[family])
        if seed_dir is None:
            print(f"  ! no valid seed found in {rdir}  — skipping")
            continue

        metric_col = METRIC_COL[family]
        df_curves  = pd.read_csv(seed_dir / "rbm_training_curves.csv")
        val_metric = df_curves[metric_col].dropna().iloc[-1]
        print(f"  seed: {seed_dir.name}   {metric_col}={val_metric:.4f}")

        if family == "nb":
            model          = load_nb(seed_dir, device)
            nan_df, taxa   = nan_rows_nb()
            score_fn       = lambda v, d, m=model: score_nb_row(m, v, d)
        else:
            model, thresh  = load_bernoulli(seed_dir, device)
            nan_df, taxa   = nan_rows_bernoulli(thresh)
            score_fn       = lambda v, d, m=model: score_bernoulli_row(m, v, d)

        df = evaluate(nan_df, taxa, score_fn, device)
        df["family"] = family
        df["split"]  = split
        df["L"]      = L
        all_dfs.append(df)

    if not all_dfs:
        print("No results — check directory names.")
        return

    full = pd.concat(all_dfs, ignore_index=True)
    full["pattern"] = full["n_miss"].map({3: "p3_3miss", 31: "p31_31miss", 54: "p54_54miss"})

    csv_out = OUT_DIR / "split_comparison.csv"
    full.to_csv(csv_out, index=False)

    summary = (full.groupby(["family", "split", "L", "pattern", "n_miss", "n_obs"])
               .agg(n_rows=("nll", "count"),
                    nll_mean=("nll", "mean"),
                    nll_std=("nll", "std"))
               .reset_index()
               .sort_values(["family", "split", "n_miss"]))

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n--- Split strategy comparison — NaN inference summary ---")
    print(summary.to_string(index=False))
    print(f"\nRows saved: {csv_out}")

    plot_comparison(summary, FIG_DIR / "split_comparison.png")


if __name__ == "__main__":
    main()
