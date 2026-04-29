"""
visualization.py — Plotting and CSV export for RBM results
=======================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


def export_results_csv(history, W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = {"epoch": history["epoch"], "train_mse": history["train_mse"]}
    if history.get("val_mse") and history["val_mse"][0] is not None:
        cols["val_mse"] = history["val_mse"]
    if history.get("train_pll"):
        cols["train_pll"] = history["train_pll"]
    if history.get("val_pll") and history["val_pll"][0] is not None:
        cols["val_pll"] = history["val_pll"]
    if history.get("train_nll"):
        cols["train_nll"] = history["train_nll"]
    if history.get("val_nll") and history["val_nll"][0] is not None:
        cols["val_nll"] = history["val_nll"]
    if history.get("theta_mean"):
        cols["theta_mean"] = history["theta_mean"]
    if history.get("sat_mid"):
        cols["sat_lo"]  = history["sat_lo"]
        cols["sat_hi"]  = history["sat_hi"]
        cols["sat_mid"] = history["sat_mid"]
    pd.DataFrame(cols).to_csv(
        os.path.join(out_dir, "rbm_training_curves.csv"), index=False)
    pd.DataFrame(W, columns=[f"h{j}" for j in range(W.shape[1])],
                 index=taxa_cols).to_csv(
        os.path.join(out_dir, "rbm_weights.csv"))
    print(f"[CSV]  saved training curves and weights → {out_dir}/")


def plot_training_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"],
            color="steelblue", lw=1.5, label="train MSE")
    if history.get("val_mse") and history["val_mse"][0] is not None:
        ax.plot(history["epoch"], history["val_mse"],
                color="firebrick", lw=1.5, ls="--", label="val MSE")
    if history.get("train_pll"):
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["train_pll"],
                 color="steelblue", lw=1.0, ls=":", label="train PLL", alpha=0.7)
        if history.get("val_pll") and history["val_pll"][0] is not None:
            ax2.plot(history["epoch"], history["val_pll"],
                     color="firebrick", lw=1.0, ls="-.", label="val PLL", alpha=0.7)
        ax2.set_ylabel("PLL")
        ax2.legend(loc="lower right")
    if history.get("train_nll"):
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["train_nll"],
                 color="steelblue", lw=1.0, ls=":", label="train NLL", alpha=0.7)
        if history.get("val_nll") and history["val_nll"][0] is not None:
            ax2.plot(history["epoch"], history["val_nll"],
                     color="firebrick", lw=1.0, ls="-.", label="val NLL", alpha=0.7)
        ax2.set_ylabel("NLL")
        ax2.legend(loc="lower right")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Training curves"); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = os.path.join(out_dir, f"training_curves{ext}")
        plt.savefig(path, dpi=150 if ext == ".png" else 300, bbox_inches="tight")
        print(f"[Plot]  saved {path}")
    plt.close()


def plot_weight_heatmap(W, taxa_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    link  = linkage(W, method="ward")
    order = leaves_list(link)
    W_ord = W[order]
    vmax  = np.abs(W).max()
    fig, ax = plt.subplots(1, 1, figsize=(8, max(5, len(W_ord)*0.15)))
    im    = ax.imshow(W_ord, aspect="auto", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"h{j}" for j in range(W.shape[1])], fontsize=9)
    ax.set_yticks(range(len(W_ord)))
    ax.set_yticklabels([taxa_cols[i] for i in order], fontsize=4)
    ax.set_title("Weight matrix W (taxa clustered)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        path = os.path.join(out_dir, f"weight_heatmap{ext}")
        plt.savefig(path, dpi=150 if ext == ".png" else 300, bbox_inches="tight")
        print(f"[Plot]  saved {path}")
    plt.close()


def plot_hidden_activations(rbm, X_train, X_val,
                            dates_train, dates_val, out_dir):
    import matplotlib.dates as mdates
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        H_all  = torch.cat([rbm.hidden_probs(X_train),
                             rbm.hidden_probs(X_val)], dim=0).cpu().numpy()
    dates_all = pd.concat([dates_train, dates_val]).reset_index(drop=True)
    n_hidden = H_all.shape[1]
    colors   = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    fig, axes = plt.subplots(n_hidden, 1,
                              figsize=(13, 2.5 * n_hidden), sharex=True)
    if n_hidden == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        vals   = H_all[:, j]
        near_0 = (vals < 0.1).mean()
        near_1 = (vals > 0.9).mean()
        mid    = 1 - near_0 - near_1
        ax.plot(dates_all, vals, lw=0.5, color=colors[j % len(colors)],
                alpha=0.6)
        ax.plot(dates_all,
                pd.Series(vals).rolling(14, center=True).mean(),
                lw=1.8, color=colors[j % len(colors)])
        ax.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(h=1|v)", fontsize=8)
        ax.set_title(
            f"h{j}  |  <0.1: {near_0:.0%}   >0.9: {near_1:.0%}   "
            f"middle: {mid:.0%}  →  "
            f"{'binary ✓' if mid < 0.15 else 'continuous'}",
            fontsize=8, loc="left"
        )
        ax.grid(True, alpha=0.25)
        for year in range(2019, 2025):
            ax.axvspan(pd.Timestamp(f"{year}-06-01"),
                       pd.Timestamp(f"{year}-09-01"),
                       alpha=0.07, color="orange")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.suptitle("Hidden unit activations h(t)  |  orange = summer",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "hidden_activations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot]  saved {path}")
    df_h = pd.DataFrame(H_all, columns=[f"h{j}" for j in range(n_hidden)])
    df_h.insert(0, "date", dates_all.values)
    df_h.to_csv(os.path.join(out_dir, "rbm_hidden_activations.csv"),
                index=False)
    print(f"[CSV]  saved hidden activations → {out_dir}/")
