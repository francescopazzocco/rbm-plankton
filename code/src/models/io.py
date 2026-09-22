"""
io.py - File I/O for the RBM plankton project.

Locations come from paths.py; nothing here re-derives the repo root.

Preprocessing:
  partition_rows      raw CSV -> (clean, nan_rows, raw, taxa); the one filter
  load_and_binarise   Bernoulli-Bernoulli RBM training tensors
  load_raw_counts     NB-Bernoulli RBM training tensors

Held-out NaN rows (called by the diagnostic scripts):
  load_nan_rows       rows _base_load keeps out of both train and val
  scale_counts        count-family view of those rows
  binarise_rows       Bernoulli-family view, using the run's own thresholds

Run navigation (CHRONO / SHUFFLED / run_dir come from paths.py, re-exported here):
  discover_run_dirs   scan RUNS_ROOT for every (family, L) of one split
  METRIC_COL          canonical val metric column per model family
  best_seed_dir       best-converged seed directory for a (family, L) run

Model loading:
  load_model          weights.npz -> the model instance that produced it
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bernoulli_rbm import BernoulliRBM
from .nb_rbm import NB_RBM, NB_ReLU_RBM, NBSigmoidRBM, NBSoftmaxRBM
from .paths import (  # noqa: F401 — re-exported so scripts have one import site
    CHRONO,
    DATA_PATH,
    RUNS_ROOT,
    SHUFFLED,
    SPLITS,
    run_dir,
    split_out_dir,
    split_suffix,
)
from .utils import load_weights
from .zinb_rbm import ZINB_RBM, ZINB_ReLU_RBM, ZINBSigmoidRBM, ZINBSoftmaxRBM

VAL_FRAC = 0.15

# Multiplier applied to organisms/uL before training; mirrors config.COUNT_SCALE.
COUNT_SCALE = 1000


# -- Training data -------------------------------------------------------------

def partition_rows(path=DATA_PATH):
    """Partition the raw CSV into the three row sets every consumer starts from.

    Steps, in order: parse, sort by date, drop all-zero rows (instrument
    downtime), then separate the rows carrying at least one NaN taxon.

    Returns (clean, nan_rows, raw, taxa_cols).  clean feeds train/val; nan_rows
    is the held-out imputation test set (LOG-002); raw is kept for the
    exploratory NaN-structure figures.  Every loader here and in
    dataset_analysis.py goes through this function so the filters cannot drift.
    """
    raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    taxa_cols = [c for c in raw.columns if c != "date"]

    nonzero = raw[raw[taxa_cols].fillna(0).sum(axis=1) > 0].copy().reset_index(drop=True)

    nan_mask = nonzero[taxa_cols].isna().any(axis=1)
    nan_rows = nonzero[nan_mask].copy().reset_index(drop=True)
    clean    = nonzero[~nan_mask].copy().reset_index(drop=True)

    return clean, nan_rows, raw, taxa_cols


def _base_load(path, val_frac, device, shuffle=False):
    """Shared steps: parse, sort, drop zero rows, separate NaN rows, split."""
    df, nan_rows, _, taxa_cols = partition_rows(path)

    print(f"[Data]  clean rows: {len(df)}  |  NaN test rows: {len(nan_rows)}")

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
        print("[Split] shuffled before split")

    n_val   = int(len(df) * val_frac)
    n_train = len(df) - n_val
    dates_train = df["date"].iloc[:n_train]
    dates_val   = df["date"].iloc[n_train:]
    print(f"[Split] train: {n_train}  val: {n_val}  "
          f"(val starts {dates_val.iloc[0].date()})")

    return df, taxa_cols, dates_train, dates_val, nan_rows, n_train


def load_and_binarise(path=DATA_PATH, binarize="median", scale=COUNT_SCALE,
                      val_frac=VAL_FRAC, device=torch.device("cpu"),
                      shuffle=False):
    """
    Preprocessing for Bernoulli-Bernoulli RBM.
    Returns binary tensors in {0, 1}.

    scale is applied before binarisation for unit consistency with
    load_raw_counts (organisms/μL -> organisms/mL).  Binarisation is
    rank-invariant under positive scaling so the binary output is identical
    to scale=1; only the stored thresholds change units accordingly.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device, shuffle=shuffle)

    X = df[taxa_cols].values.astype(np.float32) * scale
    print(f"[Counts] scale={scale}  range=[{X.min():.4f}, {X.max():.4f}]")

    if binarize == "zero":
        thresholds = np.zeros(len(taxa_cols), dtype=np.float32)
        X_bin = (X > 0).astype(np.float32)
    elif binarize == "median":
        thresholds = np.median(X, axis=0)
        X_bin = (X > thresholds).astype(np.float32)
    else:
        raise ValueError(f"Unknown binarize='{binarize}'.")

    n_const = ((X_bin.mean(0) < 0.02) | (X_bin.mean(0) > 0.98)).sum()
    if n_const > 0:
        print(f"[Warning] {n_const} taxa >98% constant after binarisation")
    print(f"[Binarise] threshold='{binarize}'")

    X_train = torch.tensor(X_bin[:n_train],  dtype=torch.float32, device=device)
    X_val   = torch.tensor(X_bin[n_train:],  dtype=torch.float32, device=device)
    return X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds


def load_raw_counts(path=DATA_PATH, scale=COUNT_SCALE,
                    val_frac=VAL_FRAC, device=torch.device("cpu"),
                    shuffle=False):
    """
    Preprocessing for NB-Bernoulli RBM.
    Returns raw count concentrations (organisms/μL), optionally scaled.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device, shuffle=shuffle)

    X = df[taxa_cols].values.astype(np.float32) * scale

    print(f"[Counts] scale={scale}  range=[{X.min():.4f}, {X.max():.4f}]  "
          f"zeros={(X==0).mean():.1%}")

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val   = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    return X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows


# -- Held-out NaN rows ---------------------------------------------------------

def load_nan_rows(path=DATA_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Rows with at least one missing taxon, in organisms/uL.

    _base_load removes these before the train/val split, so they are unseen by
    every model regardless of split strategy (LOG-002).  Returns (rows, taxa).
    """
    _, nan_rows, _, taxa_cols = partition_rows(path)
    return nan_rows, taxa_cols


def scale_counts(rows: pd.DataFrame, taxa: list[str],
                 scale: float = COUNT_SCALE) -> pd.DataFrame:
    """Count-family view of load_nan_rows output: same scaling as load_raw_counts."""
    out = rows.copy()
    out[taxa] = out[taxa] * scale
    return out


def binarise_rows(rows: pd.DataFrame, taxa: list[str],
                  thresholds: np.ndarray) -> pd.DataFrame:
    """Bernoulli-family view of load_nan_rows output, NaN preserved as NaN.

    thresholds must come from the run's own weights.npz, never recomputed from
    the data (LOG-024: bernoulli_zero stores a threshold of 0, which no median
    reproduces).

    UNIT HAZARD, unresolved: rows are in organisms/uL and are compared to the
    stored thresholds as-is.  That is correct for the Bernoulli runs currently in
    training_runs/, whose thresholds are also in organisms/uL — they were trained
    before COUNT_SCALE reached load_and_binarise.  Retraining a Bernoulli family
    with today's train.py stores thresholds multiplied by COUNT_SCALE, and this
    comparison would then binarise almost everything to 0.  Binarisation is
    rank-invariant under scaling (LOG-024) only when data and thresholds share
    units.  The warning below catches the mismatch instead of scoring silently;
    picking the fix belongs to Phase B, which owns the scale-invariance test
    (B-3 item 6 in .claude/REORG_AND_VALIDATION.md).
    """
    X = rows[taxa].values.astype(np.float32)

    if np.nanmax(X) < np.median(thresholds):
        print("[Warning] every threshold exceeds the data median — thresholds and "
              "rows are probably in different units (see binarise_rows docstring)")

    X_bin = (X > thresholds).astype(np.float32)
    X_bin[np.isnan(X)] = np.nan
    out = rows.copy()
    out[taxa] = X_bin
    return out


# -- Run navigation ------------------------------------------------------------

def discover_run_dirs(runs_root: Path = RUNS_ROOT,
                      split: str = CHRONO) -> dict[str, dict[int, list[Path]]]:
    """Scan runs_root for {family}_L{n}{split}/seed_* directories.

    Returns {family: {L: [seed_dir_paths]}} sorted by family name and L value.
    """
    suffix = split_suffix(split)
    pattern = re.compile(rf"^(.+)_L(\d+){re.escape(suffix)}$")
    runs: dict[str, dict[int, list[Path]]] = {}
    for d in sorted(Path(runs_root).iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        seed_dirs = sorted(d.glob("seed_*"))
        if seed_dirs:
            runs.setdefault(family, {})[l_val] = seed_dirs
    return runs


def load_hidden_activations(csv_path: Path, indexed: bool = True) -> pd.DataFrame:
    """Load rbm_hidden_activations.csv, keeping only the h* columns.

    indexed=True  -> date-indexed frame of hidden units (default).
    indexed=False -> 'date' kept as a column, rows sorted by date.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    hidden_cols = [c for c in df.columns if c.startswith("h")]
    if not hidden_cols:
        raise ValueError(f"No hidden-unit columns found in {csv_path}")
    if indexed:
        return df.set_index("date")[hidden_cols]
    return df[["date", *hidden_cols]].reset_index(drop=True)


METRIC_COL = {
    "nb":               "val_nll",
    "nb_relu":          "val_nll",
    "nb_sigmoid":       "val_nll",
    "nb_softmax":       "val_nll",
    "zinb":             "val_nll",
    "zinb_relu":        "val_nll",
    "zinb_sigmoid":     "val_nll",
    "zinb_softmax":     "val_nll",
    "bernoulli_median": "val_pll",
    "bernoulli_zero":   "val_pll",
}


def best_seed_dir(family_l_dir: Path, metric_col: str) -> Path | None:
    """Return the seed_* subdir with the lowest final val metric."""
    best_val, best_dir = float("inf"), None
    for seed_dir in sorted(Path(family_l_dir).glob("seed_*")):
        csv = seed_dir / "rbm_training_curves.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if metric_col not in df.columns:
            continue
        series = df[metric_col].dropna()
        if series.empty:
            continue
        v = series.iloc[-1]
        if v < best_val:
            best_val, best_dir = v, seed_dir
    return best_dir


# -- Model loading -------------------------------------------------------------

# Family -> class, matching the _MODEL_REGISTRY that train.py saves under
# npz["visible_model"].  ALL_FAMILIES in _constants.py cannot hold this map:
# the model modules import _constants, so it would be a circular import.
_MODEL_CLASSES = {
    "bernoulli_median": BernoulliRBM,
    "bernoulli_zero":   BernoulliRBM,
    "nb":               NB_RBM,
    "nb_relu":          NB_ReLU_RBM,
    "nb_sigmoid":       NBSigmoidRBM,
    "nb_softmax":       NBSoftmaxRBM,
    "zinb":             ZINB_RBM,
    "zinb_relu":        ZINB_ReLU_RBM,
    "zinb_sigmoid":     ZINBSigmoidRBM,
    "zinb_softmax":     ZINBSoftmaxRBM,
}


def load_model(seed_dir: Path, device=torch.device("cpu"), family: str | None = None):
    """Rebuild the model that produced seed_dir/weights.npz.

    The family is read from npz["visible_model"], written by train.py; pass
    family explicitly only for runs saved before that key existed.

    Returns (model, npz).  npz also carries "taxa" and, for the Bernoulli
    families, the "thresholds" that binarised the training data — callers must
    read thresholds from there rather than recomputing them (LOG-024).
    """
    npz = load_weights(Path(seed_dir) / "weights.npz")

    if family is None:
        if "visible_model" not in npz.files:
            raise KeyError(
                f"{seed_dir}/weights.npz has no 'visible_model' key; "
                f"pass family= explicitly.")
        family = str(npz["visible_model"].item())

    if family not in _MODEL_CLASSES:
        raise ValueError(f"Unknown family {family!r}; expected one of "
                         f"{sorted(_MODEL_CLASSES)}.")

    def as_tensor(key):
        return torch.tensor(npz[key], dtype=torch.float32, device=device)

    n_visible, n_hidden = npz["W"].shape
    model = _MODEL_CLASSES[family](n_visible, n_hidden, device=device)
    model.W, model.a, model.b = as_tensor("W"), as_tensor("a"), as_tensor("b")
    if "log_theta" in npz.files:
        model.log_theta = as_tensor("log_theta")
    if "logit_pi" in npz.files:
        model.logit_pi = as_tensor("logit_pi")
    return model, npz
