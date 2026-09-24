"""
use_trained_rbm.py

Load a trained RBM from a weights.npz and instantiate the corresponding
model class from `models` so you can evaluate or inspect it.

Example usage:
  conda activate Vision
  python scripts/use_trained_rbm.py --weights artifacts/models/nb/chrono/L6/seed_0/weights.npz

Or specify family+seed:
  python scripts/use_trained_rbm.py --family nb_L6 --seed 0

Options:
    --weights PATH     explicit path to weights.npz (overrides --family/--seed)
    --family NAME      '<family>_L<n>[_shuffled]', e.g. nb_L6 (use with --seed);
                       resolved under artifacts/models/<family>/<split>/L<n>/
    --seed INT         seed index (0-based) or folder name (seed_0, use with --family)
    --CD [N]           run N chained reconstruction steps (default: 5)
    --progressive      plot per-step traces for reconstruction error and distance
    --device DEVICE    'cpu' or 'cuda' (default: auto)
    --print-n-summary  print a small numeric summary (nll/pll) on train set
    --overall          every family x L=3..8 x seed (use with --shuffle), top-3 per metric
"""

import argparse
import importlib
import re
from pathlib import Path

import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from models import utils as model_utils
from models import io as data_io
from models.palette import get_palette
from models.paths import CHRONO, DIAGNOSTIC_ROOT, SHUFFLED, model_dir

_FAMILY_DIR_RE = re.compile(r"^(?P<family>.+)_L(?P<n_hidden>\d+)(?P<shuffled>_shuffled)?$")


def parse_family_dir(name: str) -> tuple[str, int, str]:
    """Parse a legacy '<family>_L<n>[_shuffled]' --family value into (family, L, split)."""
    m = _FAMILY_DIR_RE.match(name)
    if not m:
        raise ValueError(f"--family {name!r} doesn't match '<family>_L<n>[_shuffled]'")
    split = SHUFFLED if m.group("shuffled") else CHRONO
    return m.group("family"), int(m.group("n_hidden")), split

# NOTE: instantiate_model_from_weights below is a second implementation of what
# models.io.load_model now does.  Retiring it (A-3 in .claude/REORG_AND_VALIDATION.md)
# is blocked: compare_model_reconstructions.py imports this module, and that
# script is being reviewed before it is touched.

FAMILY_CLASS_MAP = {
    # bernoulli
    'bernoulli': ('bernoulli_rbm', 'BernoulliRBM'),
    'bernoulli_median': ('bernoulli_rbm', 'BernoulliRBM'),
    'bernoulli_zero': ('bernoulli_rbm', 'BernoulliRBM'),
    # nb variants
    'nb': ('nb_rbm', 'NB_RBM'),
    'nb_relu': ('nb_rbm', 'NB_ReLU_RBM'),
    'nb_sigmoid': ('nb_rbm', 'NBSigmoidRBM'),
    'nb_softmax': ('nb_rbm', 'NBSoftmaxRBM'),
    # zinb variants
    'zinb': ('zinb_rbm', 'ZINB_RBM'),
    'zinb_relu': ('zinb_rbm', 'ZINB_ReLU_RBM'),
    'zinb_sigmoid': ('zinb_rbm', 'ZINBSigmoidRBM'),
    'zinb_softmax': ('zinb_rbm', 'ZINBSoftmaxRBM'),
}


def infer_module_class(visible_model: str, override_class: str | None = None):
    if override_class:
        # try to find module automatically by scanning known modules
        # fallback: user provided full class name as Module:Class or Class
        if ':' in override_class:
            mod, cls = override_class.split(':', 1)
            return mod, cls
        return None, override_class

    key = visible_model
    # Some saved visible_model strings include the full family like 'nb_relu'
    if key in FAMILY_CLASS_MAP:
        return FAMILY_CLASS_MAP[key]
    # fallback heuristics
    if key.startswith('nb'):
        if 'relu' in key:
            return ('nb_rbm', 'NB_ReLU_RBM')
        if 'sigmoid' in key:
            return ('nb_rbm', 'NBSigmoidRBM')
        if 'softmax' in key:
            return ('nb_rbm', 'NBSoftmaxRBM')
        return ('nb_rbm', 'NB_RBM')
    if key.startswith('zinb'):
        if 'relu' in key:
            return ('zinb_rbm', 'ZINB_ReLU_RBM')
        if 'sigmoid' in key:
            return ('zinb_rbm', 'ZINBSigmoidRBM')
        if 'softmax' in key:
            return ('zinb_rbm', 'ZINBSoftmaxRBM')
        return ('zinb_rbm', 'ZINB_RBM')
    if key.startswith('bernoulli'):
        return ('bernoulli_rbm', 'BernoulliRBM')

    raise ValueError(f"Cannot infer model class from visible_model='{visible_model}'")


def load_weights_npz(path: Path):
    npz = np.load(path, allow_pickle=True)
    return npz


def instantiate_model_from_weights(npz, device_str='cpu', override_class=None):
    # get visible_model saved during training
    visible_model = None
    if 'visible_model' in npz:
        visible_model = str(npz['visible_model'].tolist())
    else:
        # try to infer from filename or require override
        raise ValueError("weights.npz missing 'visible_model' key; provide --class-name")

    mod_name, class_name = infer_module_class(visible_model, override_class)

    if mod_name is None:
        # class_name only provided - attempt to find in known modules
        candidates = ['nb_rbm', 'zinb_rbm', 'bernoulli_rbm']
        found = None
        for c in candidates:
            try:
                m = importlib.import_module(f"models.{c}")
                if hasattr(m, class_name):
                    mod_name = c
                    found = True
                    break
            except Exception:
                continue
        if mod_name is None:
            raise ImportError(f"Could not locate class {class_name} in models")

    # dynamic import
    mod = importlib.import_module(f"models.{mod_name}")
    cls = getattr(mod, class_name)

    # locate weight arrays
    if 'W' not in npz:
        raise ValueError("weights file doesn't contain 'W' array")
    W = npz['W']
    a = npz['a']
    b = npz['b']

    n_visible, n_hidden = W.shape

    device = torch.device('cuda' if (device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # Instantiate with reasonable defaults when extra args required
    kwargs = {}
    if mod_name in ('nb_rbm', 'zinb_rbm'):
        kwargs['theta_init_log'] = 0.0

    model = cls(n_visible=n_visible, n_hidden=n_hidden, device=device, **kwargs)

    # Overwrite parameters from saved arrays
    model.W = torch.tensor(W, dtype=torch.float32, device=device)
    model.a = torch.tensor(a, dtype=torch.float32, device=device)
    model.b = torch.tensor(b, dtype=torch.float32, device=device)

    # optional params
    if 'log_theta' in npz:
        model.log_theta = torch.tensor(npz['log_theta'], dtype=torch.float32, device=device)
    if 'logit_pi' in npz:
        model.logit_pi = torch.tensor(npz['logit_pi'], dtype=torch.float32, device=device)
    if 'thresholds' in npz:
        # thresholds are numpy array of floats
        model.thresholds = npz['thresholds']

    return model, visible_model


def reconstruct_chain(model, visible, steps: int):
    """Apply `steps` consecutive reconstruction passes."""
    current = visible
    for _ in range(steps):
        with torch.no_grad():
            current = model.reconstruct(current.unsqueeze(0)).squeeze(0)
    return current


def reconstruct_chain_progressive(model, visible, steps: int):
    """Apply reconstruction passes and keep the intermediate states."""
    states = []
    current = visible
    for _ in range(steps):
        with torch.no_grad():
            current = model.reconstruct(current.unsqueeze(0)).squeeze(0)
        states.append(current)
    return states


# --overall: every family x L x seed on the shuffled split, mask-one-species,
# then the top 3 (family, L) per metric (mean over seeds +- SEM across seeds).
OVERALL_FAMILIES = [
    "bernoulli_median", "bernoulli_zero",
    "nb", "nb_relu", "nb_sigmoid", "nb_softmax",
    "zinb", "zinb_relu", "zinb_sigmoid", "zinb_softmax",
]
OVERALL_HIDDEN_UNITS = range(3, 9)
OVERALL_SEEDS = range(10)
OVERALL_METRICS = (
    # (key, title, ylabel, higher_is_better)
    ("reconstruction_error", "Reconstruction Error (masked entry)", "L1 error", False),
    ("reconstructed_distance", "Reconstructed Error (other entries)", "Mean L1 error", False),
    ("cosine_other", "Cosine Similarity (other entries)", "Cosine similarity", True),
    # Robustness check: error on log(1+x), i.e. a fold error for abundant taxa
    # (0.69 = factor 2) and ~ the absolute error for rare ones, so every taxon
    # weighs the same whatever its abundance.
    ("log_reconstruction_error", "Log Error (masked entry)", "|log(1+x) - log(1+x_hat)|", False),
    ("log_reconstructed_distance", "Log Error (other entries)", "Mean |log(1+x) - log(1+x_hat)|", False),
)

METRIC_KEYS = ('reconstruction_error', 'reconstructed_distance', 'cosine_masked', 'cosine_other',
               'log_reconstruction_error', 'log_reconstructed_distance')


def load_evaluation_split(visible_model: str, device: torch.device, shuffle: bool,
                          seed: int | None = None):
    """Validation rows of one trained run.

    The shuffled split is drawn with numpy's global RNG, which train.py seeds
    with the run's seed right before loading.  Re-seeding here gives back that
    run's own held-out rows; an unseeded draw would be ~85% training rows.
    """
    if shuffle and seed is not None:
        np.random.seed(seed)
    if visible_model.startswith('bernoulli'):
        _, X_val, _, _, taxa_cols, _, _ = data_io.load_and_binarise(device=device, shuffle=shuffle)
    else:
        _, X_val, _, _, taxa_cols, _ = data_io.load_raw_counts(device=device, shuffle=shuffle)
    return X_val.to(device), list(taxa_cols)


def _row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity, 0 where either row is all zeros."""
    denom = a.norm(dim=1) * b.norm(dim=1)
    dot = (a * b).sum(dim=1)
    return torch.where(denom > 0, dot / denom.clamp_min(1e-30), torch.zeros_like(dot))


def _mean_sem(x: torch.Tensor) -> tuple[float, float]:
    x = x.double()
    return float(x.mean()), float(x.std(unbiased=False) / np.sqrt(x.numel()))


@torch.no_grad()
def evaluate_species_metrics(model, X_val: torch.Tensor, reps: int, cd: int | None):
    """Mask-one-species evaluation, same metrics as the single-run mode.

    For each species i every test row is repeated `reps` times, entry i is set
    to 0 and the row is reconstructed (cd chained passes, or one).  All rows of
    one species go through the model as one batch; each row samples its own h,
    so this is the same estimator as looping row by row.
    """
    N, D = X_val.shape
    base = X_val.repeat_interleave(reps, dim=0)
    out = {key: {'means': [], 'sems': []} for key in METRIC_KEYS}
    log_base = torch.log1p(base)
    for i in range(D):
        V = base.clone()
        V[:, i] = 0.0
        rec = V
        for _ in range(cd or 1):
            rec = model.reconstruct(rec)
        mask = torch.ones(D, dtype=torch.bool, device=X_val.device)
        mask[i] = False
        per_row = {
            'reconstruction_error': (base[:, i] - rec[:, i]).abs(),
            'reconstructed_distance': (base[:, mask] - rec[:, mask]).abs().mean(dim=1),
            'cosine_masked': _row_cosine(base[:, i:i + 1], rec[:, i:i + 1]),
            'cosine_other': _row_cosine(base[:, mask], rec[:, mask]),
            'log_reconstruction_error': (log_base[:, i] - torch.log1p(rec[:, i])).abs(),
            'log_reconstructed_distance': (log_base[:, mask] - torch.log1p(rec[:, mask])).abs().mean(dim=1),
        }
        for key, vals in per_row.items():
            m, se = _mean_sem(vals)
            out[key]['means'].append(m)
            out[key]['sems'].append(se)
    for bundle in out.values():
        bundle['means'] = np.asarray(bundle['means'])
        bundle['sems'] = np.asarray(bundle['sems'])
        bundle['score'] = float(bundle['means'].sum())
    return out


def aggregate_family_l_results(run_entries: list[dict]):
    """Mean over seeds per species, SEM across seeds; score = sum over species."""
    aggregated = []
    for entry in run_entries:
        seed_entries = entry['seed_entries']
        metric_bundle = {}
        for metric_key, _, _, higher_is_better in OVERALL_METRICS:
            stacked = np.stack([se['metrics'][metric_key]['means'] for se in seed_entries], axis=0)
            mean = stacked.mean(axis=0)
            sem = (stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
                   if stacked.shape[0] > 1 else np.zeros_like(mean))
            metric_bundle[metric_key] = {'means': mean, 'sems': sem, 'score': float(mean.sum()),
                                         'higher_is_better': higher_is_better}
        aggregated.append({**entry, 'seed_count': len(seed_entries), 'metrics': metric_bundle})
    return aggregated


def select_top_runs(aggregated_runs: list[dict], metric_key: str, k: int = 3):
    higher_is_better = next(cfg[3] for cfg in OVERALL_METRICS if cfg[0] == metric_key)
    ranked = sorted(aggregated_runs, key=lambda e: e['metrics'][metric_key]['score'],
                    reverse=higher_is_better)
    return ranked[:k]


def plot_overall_summary(selected_top: dict[str, list[dict]], taxa_cols: list[str], out_path: Path):
    fig, axes = plt.subplots(len(OVERALL_METRICS), 1, figsize=(18, 5.8 * len(OVERALL_METRICS)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = get_palette(3)
    x = np.arange(len(taxa_cols))
    for ax, (metric_key, title, ylabel, _) in zip(axes, OVERALL_METRICS):
        for idx, entry in enumerate(selected_top[metric_key]):
            m = entry['metrics'][metric_key]
            ax.errorbar(x, m['means'], yerr=m['sems'], fmt='-o', linewidth=1.5, markersize=3,
                        color=colors[idx % len(colors)],
                        label=f"{entry['family']} | L={entry['L']} (score={m['score']:.2f}, "
                              f"{entry['seed_count']} seeds)")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(taxa_cols, rotation=90, fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Overall] Saved plots to {out_path}")


def write_overall_tables(aggregated_runs: list[dict], selected_top: dict[str, list[dict]],
                         taxa_cols: list[str], out_dir: Path):
    """scores.csv: one row per (family, L); per_species.csv: long format with seed SEM;
    per_seed_scores.csv: one row per (family, L, seed); top3.json."""
    import json
    import pandas as pd
    keys = METRIC_KEYS
    rows, long_rows, seed_rows = [], [], []
    for e in aggregated_runs:
        row = {'family': e['family'], 'L': e['L'], 'seeds': e['seed_count']}
        for key in keys:
            seed_scores = np.array([se['metrics'][key]['score'] for se in e['seed_entries']])
            row[f'{key}_score_mean'] = seed_scores.mean()
            row[f'{key}_score_std'] = seed_scores.std(ddof=1) if seed_scores.size > 1 else 0.0
        rows.append(row)
        for se in e['seed_entries']:
            seed_rows.append({'family': e['family'], 'L': e['L'], 'seed': se['seed'],
                              **{f'{key}_score': se['metrics'][key]['score'] for key in keys}})
        for metric_key, m in e['metrics'].items():
            for t, mean, sem in zip(taxa_cols, m['means'], m['sems']):
                long_rows.append({'family': e['family'], 'L': e['L'], 'metric': metric_key,
                                  'taxon': t, 'mean': mean, 'sem_across_seeds': sem})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / 'scores.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / 'per_seed_scores.csv', index=False)
    pd.DataFrame(long_rows).to_csv(out_dir / 'per_species.csv', index=False)
    top = {k: [{'family': e['family'], 'L': e['L'], 'score': e['metrics'][k]['score']} for e in v]
           for k, v in selected_top.items()}
    (out_dir / 'top3.json').write_text(json.dumps(top, indent=2), encoding='utf-8')
    print(f"[Overall] Saved tables to {out_dir}")


def run_overall(args):
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    split = SHUFFLED if args.shuffle else CHRONO
    print(f'\n[Overall] mask-one-species evaluation, split={split}, reps={args.reps}, CD={args.CD}')
    records, taxa_ref = [], None
    for family in OVERALL_FAMILIES:
        for n_hidden in OVERALL_HIDDEN_UNITS:
            seed_entries = []
            for seed in OVERALL_SEEDS:
                weights_path = model_dir(family, n_hidden, split) / f'seed_{seed}' / 'weights.npz'
                if not weights_path.exists():
                    continue
                model, _ = instantiate_model_from_weights(load_weights_npz(weights_path),
                                                          device_str=args.device)
                X_val, taxa_cols = load_evaluation_split(family, device, args.shuffle, seed)
                taxa_ref = taxa_ref or taxa_cols
                metrics = evaluate_species_metrics(model, X_val, args.reps, args.CD)
                if not all(np.isfinite(m['score']) for m in metrics.values()):
                    print(f"[Overall] skip {family} L={n_hidden} seed={seed}: non-finite metrics")
                    continue
                seed_entries.append({'seed': seed, 'metrics': metrics})
            if seed_entries:
                records.append({'family': family, 'L': n_hidden, 'seed_entries': seed_entries})
                s = [se['metrics'] for se in seed_entries]
                print(f"[Overall] {family:17s} L={n_hidden} seeds={len(s):2d}  "
                      f"recon={np.mean([m['reconstruction_error']['score'] for m in s]):.3f}  "
                      f"other_L1={np.mean([m['reconstructed_distance']['score'] for m in s]):.3f}  "
                      f"cos_other={np.mean([m['cosine_other']['score'] for m in s]):.3f}  "
                      f"log_masked={np.mean([m['log_reconstruction_error']['score'] for m in s]):.3f}  "
                      f"log_other={np.mean([m['log_reconstructed_distance']['score'] for m in s]):.3f}")
    if not records:
        raise RuntimeError('No trained runs were found for overall evaluation')

    aggregated = aggregate_family_l_results(records)
    selected_top = {key: select_top_runs(aggregated, key) for key, _, _, _ in OVERALL_METRICS}
    suffix = f"_CD{args.CD}" if args.CD is not None else ""
    out_path = (Path(args.plot_out) if args.plot_out else
                DIAGNOSTIC_ROOT / 'reconstruction_plots' / 'overall' / split
                / f'use_trained_rbm_overall_reps-{args.reps}{suffix}.png')
    plot_overall_summary(selected_top, taxa_ref, out_path)
    write_overall_tables(aggregated, selected_top, taxa_ref, out_path.parent)


def main():
    epilog = (
        "Tip: Prefer specifying both --family and --seed together to select a trained run\n"
        "(this avoids needing the full --weights path).\n\n"
        "Progressive mode:\n"
        "  --CD N           chain N reconstruction steps before scoring\n"
        "  --progressive     add a third plot row with per-step traces for\n"
        "                   reconstruction error and reconstructed distance\n\n"
        "Examples:\n"
        "  python scripts/use_trained_rbm.py --family nb_L6 --seed 0 --shuffle\n\n"
        "  python scripts/use_trained_rbm.py --family nb_L6 --seed 0 --CD 5 --progressive\n\n"
        "  python scripts/use_trained_rbm.py --weights artifacts/models/nb/chrono/L6/seed_0/weights.npz --reps 100\n\n"
        "  python scripts/use_trained_rbm.py --weights artifacts/models/bernoulli_median/chrono/L6/seed_2/weights.npz --print-n-summary\n\n"
    )

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    p = argparse.ArgumentParser(
        description='Load a trained RBM and run imputation evaluation (mask-one-species).',
        epilog=epilog,
        formatter_class=HelpFormatter,
    )
    p.add_argument('--weights', type=str, help='Path to weights.npz file')
    p.add_argument('--family', type=str, help="Legacy '<family>_L<n>[_shuffled]' name, e.g. nb_L6")
    p.add_argument('--seed', type=str, help='Seed index (0) or folder name (seed_0)', default=None)
    p.add_argument('--class-name', type=str, help='Override class name or Module:Class', default=None)
    p.add_argument('--device', type=str, choices=['cpu','cuda'], default='cpu')
    p.add_argument('--print-n-summary', action='store_true', help='Compute and print a small numeric summary')
    p.add_argument('--shuffle', action='store_true', help='Use shuffled split when extracting test set')
    p.add_argument('--CD', nargs='?', const=5, type=int, default=None, help='Number of chained reconstruction steps to run')
    p.add_argument('--progressive', action='store_true', help='Plot per-step traces for CD reconstruction')
    p.add_argument('--reps', type=int, default=20, help='Number of repetitions per mask')
    p.add_argument('--overall', action='store_true', help='Evaluate every family, L=3..8 and seed, then plot the top-3 (family, L) per metric')
    p.add_argument('--plot-out', type=str, default=None, help='Output PNG path for plots')
    args = p.parse_args()

    if args.overall:
        run_overall(args)
        return

    if args.weights:
        weights_path = Path(args.weights)
    else:
        if not args.family or args.seed is None:
            p.error('Either --weights or both --family and --seed must be provided')
        seed_part = args.seed if args.seed.startswith('seed_') else f'seed_{args.seed}'
        family, n_hidden, split = parse_family_dir(args.family)
        weights_path = model_dir(family, n_hidden, split) / seed_part / 'weights.npz'

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    npz = load_weights_npz(weights_path)
    print(f"Loaded weights from {weights_path}")

    model, visible_model = instantiate_model_from_weights(npz, device_str=args.device, override_class=args.class_name)

    print(f"Loaded model: {visible_model}  |  n_visible={model.D}  n_hidden={model.L}  device={model.device}")

    # Print available attributes
    attrs = []
    for k in ('W','a','b','log_theta','logit_pi','thresholds'):
        if k in npz:
            attrs.append(k)
    print('Saved arrays:', ', '.join(attrs))

    if args.print_n_summary:
        # load data according to visible model
        vm = visible_model
        if vm.startswith('bernoulli'):
            X_train, X_val, *_ = data_io.load_and_binarise(device=model.device)
            try:
                pll = model.pll(X_train)
                print(f"Train PLL: {pll:.4f}")
            except Exception:
                print('pll not available for this model')
        else:
            X_train, X_val, *_ = data_io.load_raw_counts(device=model.device)
            try:
                nll = model.nll(X_train)
                print(f"Train NLL: {nll:.4f}")
            except Exception:
                print('nll not available for this model')

    # Generative evaluation: mask-one-species imputation
    if args.reps and args.reps > 0:
        print('\n[Eval] Running generative imputation evaluation')
        if args.CD is not None:
            print(f'[Eval] Chained reconstruction enabled: CD={args.CD}')
        vm = visible_model
        # load appropriate data split with shuffle option; re-seed so the
        # shuffled split is this run's own held-out rows (see load_evaluation_split)
        seed_match = re.fullmatch(r'seed_(\d+)', weights_path.parent.name)
        if args.shuffle and seed_match:
            np.random.seed(int(seed_match.group(1)))
        if vm.startswith('bernoulli'):
            X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows, thresholds = \
                data_io.load_and_binarise(device=model.device, shuffle=args.shuffle)
        else:
            X_train, X_val, dates_train, dates_val, taxa_cols, nan_rows = \
                data_io.load_raw_counts(device=model.device, shuffle=args.shuffle)

        # Ensure X_val on model.device
        X_val = X_val.to(model.device)

        D = model.D
        species = list(taxa_cols)

        recon_means = []
        recon_stds = []
        dist_means = []
        dist_stds = []
        progressive_recon_means = []
        progressive_recon_stds = []
        progressive_dist_means = []
        progressive_dist_stds = []
        cos_masked_means = []
        cos_masked_stds = []
        cos_other_means = []
        cos_other_stds = []

        # iterate over species (columns)
        for i in range(D):
            errs = []
            dists = []
            progressive_errs = [[] for _ in range(args.CD or 0)]
            progressive_dists = [[] for _ in range(args.CD or 0)]
            cos_masked = []
            cos_others = []
            # iterate over test samples
            for s in range(X_val.shape[0]):
                orig = X_val[s].detach().clone()
                # repeat reps times
                for _ in range(args.reps):
                    V = orig.clone()
                    V[i] = 0.0
                    step_recs = []
                    if args.CD is None:
                        with torch.no_grad():
                            rec = model.reconstruct(V.unsqueeze(0)).squeeze(0)
                    else:
                        if args.progressive:
                            step_recs = reconstruct_chain_progressive(model, V, args.CD)
                            rec = step_recs[-1]
                        else:
                            rec = reconstruct_chain(model, V, args.CD)
                    rec = rec.detach()
                    used_input = V.detach().clone()
                    rec_tensor = rec.detach().clone()
                    # scalar reconstruction error for entry i (L2 norm -> abs)
                    re = float((orig[i] - rec[i]).pow(2).item() ** 0.5)
                    # mean L1 error over the other entries
                    mask = torch.ones(D, device=model.device, dtype=torch.bool)
                    mask[i] = False
                    rd = float(torch.abs(orig[mask] - rec[mask]).mean().item())
                    masked_orig = orig[i:i+1]
                    masked_rec = rec_tensor[i:i+1]

                    denom_masked = masked_orig.norm() * masked_rec.norm()
                    if denom_masked.item() > 0:
                        cs_masked = float(torch.dot(masked_orig, masked_rec).div(denom_masked).item())
                    else:
                        cs_masked = 0.0

                    other_orig = orig[mask]
                    other_rec = rec_tensor[mask]
                    denom_other = other_orig.norm() * other_rec.norm()
                    if denom_other.item() > 0:
                        cs_other = float(torch.dot(other_orig, other_rec).div(denom_other).item())
                    else:
                        cs_other = 0.0
                    if args.progressive and args.CD is not None:
                        for step_idx, step_rec in enumerate(step_recs):
                            step_recon = step_rec.detach()
                            progressive_errs[step_idx].append(float((orig[i] - step_recon[i]).pow(2).item() ** 0.5))
                            progressive_dists[step_idx].append(float(torch.abs(orig[mask] - step_recon[mask]).mean().item()))
                    errs.append(re)
                    dists.append(rd)
                    cos_masked.append(cs_masked)
                    cos_others.append(cs_other)

            # aggregate across all samples and reps
            errs_arr = np.array(errs, dtype=float)
            dists_arr = np.array(dists, dtype=float)
            recon_means.append(errs_arr.mean() if errs_arr.size > 0 else 0.0)
            # standard error of the mean (std / sqrt(N))
            if errs_arr.size > 0:
                recon_stds.append(errs_arr.std(ddof=0) / np.sqrt(errs_arr.size))
            else:
                recon_stds.append(0.0)

            dist_means.append(dists_arr.mean() if dists_arr.size > 0 else 0.0)
            if dists_arr.size > 0:
                dist_stds.append(dists_arr.std(ddof=0) / np.sqrt(dists_arr.size))
            else:
                dist_stds.append(0.0)

            cos_masked_arr = np.array(cos_masked, dtype=float)
            cos_masked_means.append(cos_masked_arr.mean() if cos_masked_arr.size > 0 else 0.0)
            if cos_masked_arr.size > 0:
                cos_masked_stds.append(cos_masked_arr.std(ddof=0) / np.sqrt(cos_masked_arr.size))
            else:
                cos_masked_stds.append(0.0)

            cos_other_arr = np.array(cos_others, dtype=float)
            cos_other_means.append(cos_other_arr.mean() if cos_other_arr.size > 0 else 0.0)
            if cos_other_arr.size > 0:
                cos_other_stds.append(cos_other_arr.std(ddof=0) / np.sqrt(cos_other_arr.size))
            else:
                cos_other_stds.append(0.0)

            if args.progressive and args.CD is not None:
                step_recon_means = []
                step_recon_stds = []
                step_dist_means = []
                step_dist_stds = []
                for step_idx in range(args.CD):
                    step_errs_arr = np.array(progressive_errs[step_idx], dtype=float)
                    step_dists_arr = np.array(progressive_dists[step_idx], dtype=float)
                    step_recon_means.append(step_errs_arr.mean() if step_errs_arr.size > 0 else 0.0)
                    step_recon_stds.append(step_errs_arr.std(ddof=0) / np.sqrt(step_errs_arr.size) if step_errs_arr.size > 0 else 0.0)
                    step_dist_means.append(step_dists_arr.mean() if step_dists_arr.size > 0 else 0.0)
                    step_dist_stds.append(step_dists_arr.std(ddof=0) / np.sqrt(step_dists_arr.size) if step_dists_arr.size > 0 else 0.0)
                progressive_recon_means.append(step_recon_means)
                progressive_recon_stds.append(step_recon_stds)
                progressive_dist_means.append(step_dist_means)
                progressive_dist_stds.append(step_dist_stds)

            print(
                f"[Eval] species {i}/{D} ({species[i]}): "
                f"recon_mean={recon_means[-1]:.4f} recon_std={recon_stds[-1]:.4f} "
                f"dist_mean={dist_means[-1]:.4f} "
                f"cos_masked={cos_masked_means[-1]:.4f} cos_other={cos_other_means[-1]:.4f}"
            )

        # plotting
        out_path = args.plot_out
        if out_path is None:
            # derive family and seed from provided args when available, else from weights_path
            if args.family:
                fam_name = args.family
            else:
                try:
                    fam_name = weights_path.parts[-3]
                except Exception:
                    fam_name = 'family'

            if args.seed is not None:
                seed_str = args.seed if str(args.seed).startswith('seed_') else f'seed_{args.seed}'
            else:
                try:
                    seed_str = weights_path.parts[-2]
                except Exception:
                    seed_str = 'seed'

            suffix = f"_CD{args.CD}" if args.CD is not None else ""
            fname = f"{fam_name}_{seed_str}_shuffle-{int(args.shuffle)}_reps-{args.reps}{suffix}.png"
            out_dir = DIAGNOSTIC_ROOT / 'reconstruction_plots'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(out_dir / fname)
        else:
            # ensure parent directory exists for explicit path
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            if args.CD is not None:
                out_path_obj = Path(out_path)
                if not out_path_obj.stem.endswith(f"_CD{args.CD}"):
                    out_path = str(out_path_obj.with_name(f"{out_path_obj.stem}_CD{args.CD}{out_path_obj.suffix}"))

        progressive_enabled = bool(args.progressive and args.CD is not None)
        nrows = 3 if progressive_enabled else 2
        fig, axes = plt.subplots(nrows, 2, figsize=(18, 6 * nrows))
        axes = np.atleast_2d(axes)
        x = np.arange(D)
        panel_colors = get_palette(4)
        axes[0, 0].errorbar(x, recon_means, yerr=recon_stds, fmt='o', color=panel_colors[0])
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(species, rotation=90, fontsize=6)
        axes[0, 0].set_title('Reconstruction Error by Species')
        axes[0, 0].set_ylabel('L1 error')
        axes[0, 0].grid(alpha=0.5)

        axes[0, 1].errorbar(x, dist_means, yerr=dist_stds, fmt='o', color=panel_colors[1])
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(species, rotation=90, fontsize=6)
        axes[0, 1].set_title('Reconstructed Error (other entries)')
        axes[0, 1].set_ylabel('Mean L1 error')
        axes[0, 1].grid(alpha=0.5)

        axes[1, 0].errorbar(x, cos_masked_means, yerr=cos_masked_stds, fmt='o', color=panel_colors[2])
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(species, rotation=90, fontsize=6)
        axes[1, 0].set_title('Cosine Similarity (masked entry)')
        axes[1, 0].set_ylabel('Cosine similarity')
        axes[1, 0].grid(alpha=0.5)

        axes[1, 1].errorbar(x, cos_other_means, yerr=cos_other_stds, fmt='o', color=panel_colors[3])
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(species, rotation=90, fontsize=6)
        axes[1, 1].set_title('Cosine Similarity (other entries)')
        axes[1, 1].set_ylabel('Cosine similarity')
        axes[1, 1].grid(alpha=0.5)

        if progressive_enabled:
            progressive_recon_means_arr = np.array(progressive_recon_means, dtype=float)
            progressive_recon_stds_arr = np.array(progressive_recon_stds, dtype=float)
            progressive_dist_means_arr = np.array(progressive_dist_means, dtype=float)
            progressive_dist_stds_arr = np.array(progressive_dist_stds, dtype=float)

            cmap = plt.get_cmap('viridis', args.CD)
            step_labels = [f'step {idx + 1}' for idx in range(args.CD)]
            for step_idx in range(args.CD):
                color = cmap(step_idx)
                axes[2, 0].errorbar(
                    x,
                    progressive_recon_means_arr[:, step_idx],
                    yerr=progressive_recon_stds_arr[:, step_idx],
                    fmt='-o',
                    color=color,
                    linewidth=1.2,
                    markersize=3,
                    label=step_labels[step_idx],
                )
                axes[2, 1].errorbar(
                    x,
                    progressive_dist_means_arr[:, step_idx],
                    yerr=progressive_dist_stds_arr[:, step_idx],
                    fmt='-o',
                    color=color,
                    linewidth=1.2,
                    markersize=3,
                    label=step_labels[step_idx],
                )

            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(species, rotation=90, fontsize=6)
            axes[2, 0].set_title('Progressive Reconstruction Error by Species')
            axes[2, 0].set_ylabel('L1 error')
            axes[2, 0].grid(alpha=0.5)
            axes[2, 0].legend(fontsize=7, ncol=min(args.CD, 4), loc='upper right')

            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels(species, rotation=90, fontsize=6)
            axes[2, 1].set_title('Progressive Reconstructed Error (other entries)')
            axes[2, 1].set_ylabel('Mean L1 error')
            axes[2, 1].grid(alpha=0.5)
            axes[2, 1].legend(fontsize=7, ncol=min(args.CD, 4), loc='upper right')

        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"[Eval] Saved plots to {out_path}")


if __name__ == '__main__':
    main()
