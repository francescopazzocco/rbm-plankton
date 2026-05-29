import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import euclidean

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
    parser.add_argument("--weights", type=str, default="weights/zinb_sigmoid_L7_seed9_best.npz", help="npz weights path")
    parser.add_argument("--archetypes", type=str, default="Cheng/Data/archetypes_k5_profiles.csv", help="archetypes csv path")
    parser.add_argument("--out", type=str, default="analysis/results/archetype_closest_rbm.png", help="output plot path")
    parser.add_argument("--metric", type=str, choices=["euclidean", "cosine"], default="cosine", help="distance metric")
    parser.add_argument("--top_k", type=int, default=3, help="number of closest RBM hidden units to show")
    args = parser.parse_args()

    # Load Archetypes
    df_arch = pd.read_csv(args.archetypes, index_col=0)
    arch_taxa = df_arch.columns.values
    
    # Load RBM weights
    d = np.load(args.weights, allow_pickle=True)
    W = d['W']
    rbm_taxa = d['taxa']
    a = d['a'] if 'a' in d else np.zeros(W.shape[0])
    visible_model = str(d['visible_model']) if 'visible_model' in d else 'bernoulli'
    logit_pi = d['logit_pi'] if 'logit_pi' in d else None
    n_visible, n_hidden = W.shape
    
    # Align taxa
    common_taxa = np.intersect1d(arch_taxa, rbm_taxa)
    df_arch = df_arch[common_taxa]
    
    taxa_to_idx = {t: i for i, t in enumerate(rbm_taxa)}
    idx_keep = [taxa_to_idx[t] for t in common_taxa]
    
    # Normalize Archetypes (L2 norm = 1)
    A_mat = df_arch.values
    A_norms = np.linalg.norm(A_mat, axis=1, keepdims=True)
    A_norms[A_norms == 0] = 1
    A_mat_norm = A_mat / A_norms
    
    # Compute visible activations for each hidden unit (L2 norm = 1)
    W_vis = np.zeros((len(common_taxa), n_hidden))
    for h in range(n_hidden):
        W_vis[:, h] = compute_visible_activation(W, a, None, visible_model, h, logit_pi)[idx_keep]
    
    W_norms = np.linalg.norm(W_vis, axis=0, keepdims=True)
    W_norms[W_norms == 0] = 1
    W_vis_norm = W_vis / W_norms
    
    # Find 3 closest hidden units per archetype
    n_arch = A_mat.shape[0]
    arch_names = list(df_arch.index)
    
    # Create figure with subplots (one per archetype)
    fig, axes = plt.subplots(n_arch, 1, figsize=(14, 3*n_arch))
    if n_arch == 1:
        axes = [axes]
    
    # Generate colors based on top_k
    if args.top_k == 3:
        colors_hidden = ['goldenrod', 'forestgreen', 'midnightblue']
        linestyles = ['-', '--', ':']
    else:
        # Use viridis: nearest (index 0) = high value (bright), furthest (index top_k-1) = low value (dark)
        cmap = mpl.colormaps['viridis']
        colors_hidden = [cmap(1.0 - i / (args.top_k - 1) if args.top_k > 1 else 1.0) for i in range(args.top_k)]
        linestyles = ['-', '--', '-.', ':'] * ((args.top_k // 4) + 1)  # Cycle through linestyles
        linestyles = linestyles[:args.top_k]
    
    # For each archetype
    for i, arch in enumerate(arch_names):
        arch_vec = A_mat_norm[i]
        arch_vals = A_mat[i]
        
        # Find 3 closest hidden units
        distances = []
        for j in range(n_hidden):
            if args.metric == "euclidean":
                dist = euclidean(arch_vec, W_vis_norm[:, j])
            else:
                # cosine similarity (dot product of normalized vectors)
                dist = -np.dot(arch_vec, W_vis_norm[:, j])  # negate for sorting (lower = closer)
            distances.append((j, dist))
        
        distances.sort(key=lambda x: x[1])
        closest_k = [d[0] for d in distances[:args.top_k]]
        
        # Plot
        ax = axes[i]
        x_pos = np.arange(len(common_taxa))
        
        # Left y-axis: archetype weights (wide bars)
        ax.bar(x_pos, arch_vals, width=0.9, label=arch, alpha=0.6, color='dimgrey')
        
        # Right y-axis: closest RBM activations (lines with different linestyles, ordered by distance)
        ax2 = ax.twinx()
        for k_idx, h in enumerate(closest_k):
            ax2.plot(x_pos, W_vis[:, h], linestyle=linestyles[k_idx], linewidth=2, label=f"H{h}", alpha=0.8, color=colors_hidden[k_idx])
        
        ax.set_ylabel(f"{arch} Weight", color='dimgrey')
        ax2.set_ylabel(f"RBM Activation (Top {args.top_k})", color='black')
        ax.set_title(f"{arch} vs Closest RBM Hidden Units ({', '.join([f'H{h}' for h in closest_k])})")
        
        # Set xticks for all plots
        ax.set_xticks(x_pos)
        
        # Only set x-axis labels for the last subplot
        if i == n_arch - 1:
            ax.set_xticklabels(common_taxa, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel("Taxa")
        else:
            ax.set_xticklabels([])
        
        ax.tick_params(axis='y', labelcolor='dimgrey')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
