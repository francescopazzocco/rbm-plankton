import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from models.io import CHRONO, METRIC_COL, SPLITS, best_seed_dir, model_dir, split_out_dir
from models.paths import DIAGNOSTIC_ROOT, MODELS_ROOT
from models.paths import PROJECT_ROOT as ROOT


def resolve_weights(family: str, n_hidden: int, split: str, models_root: Path) -> Path:
    """Best-converged seed's weights.npz for one (family, L, split) run."""
    seed_dir = best_seed_dir(model_dir(family, n_hidden, split, models_root), METRIC_COL[family])
    if seed_dir is None:
        raise FileNotFoundError(
            f"No converged seed for {family} L={n_hidden} ({split}) in {models_root}")
    return seed_dir / "weights.npz"


def plot_heatmap(ax, matrix, annot, xticklabels, yticklabels, cmap):
    """Matplotlib-only annotated heatmap (no seaborn dependency)."""
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    thresh = matrix.min() + (matrix.max() - matrix.min()) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{annot[i, j]:d}", ha="center", va="center",
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

def get_top_pct_archetype(row, pct=0.98):
    """Get taxa making up pct of total probability mass."""
    row_sorted = row.sort_values(ascending=False)
    cum_frac = row_sorted.cumsum() / row_sorted.sum()
    # Find index where it crosses pct
    idx_pct = (cum_frac >= pct).argmax()
    return set(row_sorted.iloc[:idx_pct + 1].index)

def get_top_pct_hidden(weight_col, taxa, pct=0.98):
    """Get taxa making up pct of positive weights mass."""
    pos_mask = weight_col > 0
    w_pos = weight_col[pos_mask]
    taxa_pos = taxa[pos_mask]
    
    if len(w_pos) == 0:
        return set()
        
    idx_sort = np.argsort(w_pos)[::-1]
    w_sorted = w_pos[idx_sort]
    taxa_sorted = taxa_pos[idx_sort]
    
    cum_frac = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx_pct = (cum_frac >= pct).argmax()
    return set(taxa_sorted[:idx_pct + 1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=None,
                        help="npz weights path (default: resolved from --family/--L/--split)")
    parser.add_argument("--family", default="zinb_sigmoid", help="Model family (default: zinb_sigmoid)")
    parser.add_argument("--L", type=int, default=7, help="Hidden unit count (default: 7)")
    parser.add_argument("--split", choices=SPLITS, default=CHRONO, help="Split strategy (default: chrono)")
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--archetypes", type=Path, default=ROOT / "prof" / "archetypes_k5_profiles.csv",
                        help="archetypes csv path")
    parser.add_argument("--out", type=Path, default=None, help="output plot path")
    parser.add_argument("--pct", type=float, default=0.98, help="fraction of mass to cover")
    args = parser.parse_args()

    weights = args.weights or resolve_weights(args.family, args.L, args.split, args.models_root)
    out = args.out or (
        split_out_dir(DIAGNOSTIC_ROOT / "02_model_analysis" / "archetype" / "overlap_heatmap",
                     args.split)
        / f"{args.family}_L{args.L}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load Archetypes
    df_arch = pd.read_csv(args.archetypes, index_col=0)
    arch_sets = {arch: get_top_pct_archetype(df_arch.loc[arch], pct=args.pct) for arch in df_arch.index}

    # Load RBM weights
    d = np.load(weights, allow_pickle=True)
    W = d['W']
    a = d['a'] if 'a' in d else np.zeros(W.shape[0])
    taxa = d['taxa']
    visible_model = str(d['visible_model']) if 'visible_model' in d else 'bernoulli'
    logit_pi = d['logit_pi'] if 'logit_pi' in d else None
    n_visible, n_hidden = W.shape
    
    # Compute proper visible activations for each hidden unit
    hidden_sets = {}
    for h in range(n_hidden):
        vis_activation = compute_visible_activation(W, a, None, visible_model, h, logit_pi)
        hidden_sets[f"H{h}"] = get_top_pct_hidden(vis_activation, taxa, pct=args.pct)
    
    # Compute Intersection statistics
    res_frac = np.zeros((len(arch_sets), len(hidden_sets)))
    res_counts = np.zeros((len(arch_sets), len(hidden_sets)), dtype=int)
    arch_names = list(arch_sets.keys())
    hidden_names = list(hidden_sets.keys())
    
    for i, arch in enumerate(arch_names):
        set_A = arch_sets[arch]
        len_A = len(set_A)
        for j, h in enumerate(hidden_names):
            set_H = hidden_sets[h]
            overlap = len(set_A.intersection(set_H))
            res_counts[i, j] = overlap
            res_frac[i, j] = overlap / len_A if len_A > 0 else 0
                
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    # Heatmap colors based on fraction (res_frac), numbers printed based on count (res_counts)
    plot_heatmap(ax, res_frac, res_counts, hidden_names, arch_names, cmap="YlGnBu")
    ax.set_title(f"Archetype vs Hidden Unit Overlap\n(Color = Fraction of Archetype, Text = Common Species Count)\nTop {args.pct*100:.0f}% Mass | Weights: {weights.name}")
    ax.set_xlabel("RBM Hidden Units")
    ax.set_ylabel("Archetypes")
    fig.tight_layout()
    fig.savefig(out)
    print(f"Heatmap saved to {out}")

    # Also save CSV with fractions
    df_res = pd.DataFrame(res_frac, index=arch_names, columns=hidden_names)
    csv_out = out.with_suffix('.csv')
    df_res.to_csv(csv_out)
    print(f"CSV saved to {csv_out}")

if __name__ == "__main__":
    main()
