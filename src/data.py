"""
data.py — Data loading and preprocessing for RBM plankton models
==============================================================
"""

import numpy as np
import pandas as pd
import torch
from config import DATA_PATH, VAL_FRAC


def _base_load(path, val_frac, device):
    """Shared steps: parse, sort, drop zero rows, separate NaN rows, split."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa_cols = [c for c in df.columns if c != "date"]

    nonzero_mask = df[taxa_cols].sum(axis=1) > 0
    df = df[nonzero_mask].copy()

    nan_mask = df[taxa_cols].isna().any(axis=1)
    nan_rows = df[nan_mask].copy()
    df       = df[~nan_mask].copy().reset_index(drop=True)

    print(f"[Data]  clean rows: {len(df)}  |  NaN test rows: {len(nan_rows)}")

    n_val   = int(len(df) * val_frac)
    n_train = len(df) - n_val
    dates_train = df["date"].iloc[:n_train]
    dates_val   = df["date"].iloc[n_train:]
    print(f"[Split] train: {n_train}  val: {n_val}  "
          f"(val starts {dates_val.iloc[0].date()})")

    return df, taxa_cols, dates_train, dates_val, nan_rows, n_train


def load_and_binarise(path=DATA_PATH, binarize="median",
                      val_frac=VAL_FRAC, device=torch.device("cpu")):
    """
    Preprocessing for Bernoulli-Bernoulli RBM.
    Returns binary tensors in {0, 1}.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device)

    X = df[taxa_cols].values.astype(np.float32)

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


def load_raw_counts(path=DATA_PATH, scale=1000,
                    val_frac=VAL_FRAC, device=torch.device("cpu")):
    """
    Preprocessing for NB-Bernoulli RBM.
    Returns raw count concentrations (organisms/μL), optionally scaled.
    """
    df, taxa_cols, dates_train, dates_val, nan_rows, n_train = \
        _base_load(path, val_frac, device)

    X = df[taxa_cols].values.astype(np.float32) * scale

    print(f"[Counts] scale={scale}  range=[{X.min():.4f}, {X.max():.4f}]  "
          f"zeros={(X==0).mean():.1%}")

    X_train = torch.tensor(X[:n_train], dtype=torch.float32, device=device)
    X_val   = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
    return X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows
