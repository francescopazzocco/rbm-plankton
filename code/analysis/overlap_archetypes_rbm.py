import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument("--weights", type=str, default="weights/zinb_sigmoid_L7_seed9_best.npz", help="npz weights path")
    parser.add_argument("--archetypes", type=str, default="Cheng/Data/archetypes_k5_profiles.csv", help="archetypes csv path")
    parser.add_argument("--out", type=str, default="analysis/results/overlap_heatmap.png", help="output plot path")
    parser.add_argument("--pct", type=float, default=0.98, help="fraction of mass to cover")
    args = parser.parse_args()

    # Load Archetypes
    df_arch = pd.read_csv(args.archetypes, index_col=0)
    arch_sets = {arch: get_top_pct_archetype(df_arch.loc[arch], pct=args.pct) for arch in df_arch.index}
    
    # Load RBM weights
    d = np.load(args.weights, allow_pickle=True)
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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    # Heatmap colors based on fraction (res_frac), numbers printed based on count (res_counts)
    sns.heatmap(res_frac, annot=res_counts, cmap="YlGnBu", xticklabels=hidden_names, yticklabels=arch_names, fmt="d")
    plt.title(f"Archetype vs Hidden Unit Overlap\n(Color = Fraction of Archetype, Text = Common Species Count)\nTop {args.pct*100:.0f}% Mass | Weights: {Path(args.weights).name}")
    plt.xlabel("RBM Hidden Units")
    plt.ylabel("Archetypes")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Heatmap saved to {args.out}")
    
    # Also save CSV with fractions
    df_res = pd.DataFrame(res_frac, index=arch_names, columns=hidden_names)
    csv_out = out_path.with_suffix('.csv')
    df_res.to_csv(csv_out)
    print(f"CSV saved to {csv_out}")

if __name__ == "__main__":
    main()
