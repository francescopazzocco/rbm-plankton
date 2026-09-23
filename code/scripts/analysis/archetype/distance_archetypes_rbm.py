import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.io import CHRONO, METRIC_COL, SPLITS, best_seed_dir, model_dir, split_out_dir
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.paths import PROJECT_ROOT as ROOT
from scipy.spatial.distance import euclidean


def resolve_weights(family: str, n_hidden: int, split: str, models_root: Path) -> Path:
    """Best-converged seed's weights.npz for one (family, L, split) run."""
    seed_dir = best_seed_dir(model_dir(family, n_hidden, split, models_root), METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(
            f"No converged seed for {family} L={n_hidden} ({split}) in {models_root}")
    return seed_dir / "weights.npz"


def plot_heatmap(ax, matrix, xticklabels, yticklabels, cmap, vmin, vmax, fmt="{:.2f}"):
    """Matplotlib-only annotated heatmap (no seaborn dependency)."""
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    thresh = matrix.min() + (matrix.max() - matrix.min()) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] < thresh else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    return im


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def compute_visible_activation(W, a, b, visible_model, h, logit_pi=None):
    """Compute visible node activations when hidden unit h is on.
    
    Args:
        W: weight matrix (D, L)
        a: visible bias (D,)
        b: hidden bias (L,)
        visible_model: 'bernoulli', 'zinb', 'zinb_sigmoid', 'nb_sigmoid', etc.
        h: hidden unit index
        logit_pi: logit of zero-inflation probability for ZINB
    
    Returns:
        Visible activation vector (D,)
    """
    H_vec = np.zeros(W.shape[1])
    H_vec[h] = 1.0
    z = a + W.dot(H_vec)
    
    if 'bernoulli' in visible_model.lower():
        return sigmoid(z)
    elif 'zinb' in visible_model.lower() or 'nb' in visible_model.lower():
        mu = np.exp(np.clip(z, None, 10.0))
        if logit_pi is not None:
            pi = sigmoid(logit_pi)
            return (1.0 - pi) * mu
        return mu
    else:
        raise ValueError(f"Unknown visible model: {visible_model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="npz weights path (Model 1); default: resolved from --family/--L/--split")
    parser.add_argument("--family", default="zinb_sigmoid", help="Model family (default: zinb_sigmoid)")
    parser.add_argument("--L", type=int, default=7, help="Hidden unit count (default: 7)")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO, help="Split strategy (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--archetypes", type=Path, default=ROOT / "prof" / "archetypes_k5_profiles.csv",
                        help="archetypes csv path (used if weights2 is not provided)")
    parser.add_argument("--weights2", type=str, default=None, help="optional second npz weights path (Model 2) to compare RBM vs RBM")
    parser.add_argument("--out", type=Path, default=None, help="output plot path")
    parser.add_argument("--metric", type=str, choices=["euclidean", "cosine"], default="euclidean", help="distance metric")
    args = parser.parse_args()

    weights = args.weights or resolve_weights(args.family, args.L, args.split, args.models_root)
    out = args.out or (
        split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis" / "archetype" / "distance_heatmap",
                     args.split)
        / f"{args.family}_L{args.L}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load RBM weights 1
    d = np.load(weights, allow_pickle=True)
    W = d['W']
    rbm_taxa = d['taxa']
    a = d['a'] if 'a' in d else np.zeros(W.shape[0])
    visible_model = str(d['visible_model']) if 'visible_model' in d else 'bernoulli'
    logit_pi = d['logit_pi'] if 'logit_pi' in d else None
    n_visible, n_hidden = W.shape
    
    if args.weights2:
        # Load RBM weights 2
        d2 = np.load(args.weights2, allow_pickle=True)
        W2 = d2['W']
        a2 = d2['a'] if 'a' in d2 else np.zeros(W2.shape[0])
        arch_taxa = d2['taxa']
        visible_model2 = str(d2['visible_model']) if 'visible_model' in d2 else 'bernoulli'
        logit_pi2 = d2['logit_pi'] if 'logit_pi' in d2 else None
        n_arch_hidden = W2.shape[1]
        
        common_taxa = np.intersect1d(arch_taxa, rbm_taxa)
        
        # Compute visible activations for Model 2 (as if it were archetypes)
        W2_vis = np.zeros((len(common_taxa), n_arch_hidden))
        taxa_to_idx2 = {t: i for i, t in enumerate(arch_taxa)}
        idx_keep2 = [taxa_to_idx2[t] for t in common_taxa]
        
        for h in range(n_arch_hidden):
            W2_vis[:, h] = compute_visible_activation(W2, a2, None, visible_model2, h, logit_pi2)[idx_keep2]
        
        # Normalize W2 (Model 2)
        W2_norms = np.linalg.norm(W2_vis, axis=0, keepdims=True)
        W2_norms[W2_norms == 0] = 1
        W2_vis = W2_vis / W2_norms
        A_mat = W2_vis.T  # Transpose to match (n_arch_hidden, n_features) shape
        arch_names = [f"M2_H{h}" for h in range(n_arch_hidden)]
        y_label = "Model 2 Hidden Units"
        title_prefix = "Model 2 vs Model 1"
        
    else:
        # Load Archetypes
        df_arch = pd.read_csv(args.archetypes, index_col=0)
        arch_taxa = df_arch.columns.values
        common_taxa = np.intersect1d(arch_taxa, rbm_taxa)
        df_arch = df_arch[common_taxa]
        
        # Normalize Archetypes (L2 norm = 1)
        A_mat = df_arch.values
        A_norms = np.linalg.norm(A_mat, axis=1, keepdims=True)
        A_norms[A_norms == 0] = 1
        A_mat = A_mat / A_norms
        arch_names = list(df_arch.index)
        y_label = "Archetypes"
        title_prefix = "Archetype vs Hidden Unit"

    # Align W (Model 1) and a
    taxa_to_idx = {t: i for i, t in enumerate(rbm_taxa)}
    idx_keep = [taxa_to_idx[t] for t in common_taxa]

    # Compute visible activations for each hidden unit (Model 1)
    W_vis = np.zeros((len(common_taxa), n_hidden))
    for h in range(n_hidden):
        W_vis[:, h] = compute_visible_activation(W, a, None, visible_model, h, logit_pi)[idx_keep]
    
    # Normalize RBM Hidden Nodes (L2 norm = 1)
    W_norms = np.linalg.norm(W_vis, axis=0, keepdims=True)
    W_norms[W_norms == 0] = 1
    W_vis = W_vis / W_norms
    
    # Compute metric
    n_arch = A_mat.shape[0]
    res_metric = np.zeros((n_arch, n_hidden))
    hidden_names = [f"M1_H{h}" if args.weights2 else f"H{h}" for h in range(n_hidden)]
    
    for i in range(n_arch):
        for j in range(n_hidden):
            if args.metric == "euclidean":
                res_metric[i, j] = euclidean(A_mat[i], W_vis[:, j])
            else:
                # cosine similarity between normalized vectors is dot product
                res_metric[i, j] = np.dot(A_mat[i], W_vis[:, j])
                
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    if args.metric == "euclidean":
        cmap = "cividis_r"  # Reverse so low distance (close) is darker/yellow depending on cividis
        title = f"{title_prefix} Euclidean Distance\n(L2 Normalized Vectors)"
        vmin, vmax = None, None
    else:
        cmap = "cividis"
        title = f"{title_prefix} Cosine Similarity\n(L2 Normalized Vectors)"
        vmin, vmax = 0, 1

    plot_heatmap(ax, res_metric, hidden_names, arch_names, cmap, vmin, vmax)
    ax.set_title(title)
    ax.set_xlabel("Model 1 Hidden Units" if args.weights2 else "RBM Hidden Units")
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Heatmap saved to {out}")

    # Save CSV
    df_res = pd.DataFrame(res_metric, index=arch_names, columns=hidden_names)
    csv_out = out.with_suffix('.csv')
    df_res.to_csv(csv_out)
    print(f"CSV saved to {csv_out}")

if __name__ == "__main__":
    main()