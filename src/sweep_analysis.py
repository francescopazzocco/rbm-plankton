"""
sweep_analysis.py — NLL/PLL vs L sweep analysis across all trained models.

Reads results/{family}_L{n}/rbm_training_curves.csv for all available runs and
produces three figures:
  1. Final val metric vs L per model family
  2. Val metric training curves overlaid per family
  3. NB-specific: NLL and theta_mean trajectories per L (instability diagnostic)
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results/multiseed_pcd"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "sweep"

# Metric column and direction per family (higher/lower is better)
FAMILY_META = {
    "bernoulli_median": dict(col="val_pll", label="PLL (↑)",   better="higher"),
    "bernoulli_zero":   dict(col="val_pll", label="PLL (↑)",   better="higher"),
    "nb":               dict(col="val_nll", label="NLL (↓)",   better="lower"),
}

COLORS = {
    "bernoulli_median": "#1f77b4",
    "bernoulli_zero":   "#ff7f0e",
    "nb":               "#2ca02c",
}


def discover_runs(results_dir: Path) -> dict[str, dict[int, list[Path]]]:
    """Return {family: {L: [csv_paths]}} for all valid result directories."""
    pattern = re.compile(r"^(.+)_L(\d+)$")
    runs: dict[str, dict[int, list[Path]]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        family, l_val = m.group(1), int(m.group(2))
        csvs = list(d.glob("seed_*/rbm_training_curves.csv"))
        if csvs:
            runs.setdefault(family, {})[l_val] = sorted(csvs)
    return runs


def load_curves(csv_path: Path, col: str) -> pd.Series | None:
    """Load a metric column from a training curves CSV; return None if all NaN."""
    df = pd.read_csv(csv_path)
    if col not in df.columns or df[col].isna().all():
        return None
    return df.set_index("epoch")[col]


def aggregate_curves(csv_paths: list[Path], col: str) -> tuple[pd.Series, pd.Series] | None:
    """Aggregate a metric across seeds; return (mean, std) or None if all failed."""
    curves = [load_curves(p, col) for p in csv_paths]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    df = pd.concat(curves, axis=1)
    return df.mean(axis=1), df.std(axis=1)


def plot_final_metric(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(14, 4), sharey=False)
    fig.suptitle("Final val metric vs L (last epoch, mean ± std over seeds)", fontsize=13)

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        xs, means, stds = [], [], []
        for l_val in sorted(family_runs):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is None:
                continue
            mean_curve, std_curve = agg
            xs.append(l_val)
            means.append(mean_curve.iloc[-1])
            stds.append(std_curve.iloc[-1])

        color = COLORS[family]
        ax.errorbar(xs, means, yerr=stds, fmt="o-", color=color, linewidth=2,
                    markersize=7, capsize=4, label=family)
        for x, y, s in zip(xs, means, stds):
            ax.annotate(f"{y:.3f}±{s:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

        ax.set_title(family)
        ax.set_xlabel("L (hidden units)")
        ax.set_ylabel(meta["label"])
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_final_metric.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_training_curves(runs, figures_dir: Path):
    fig, axes = plt.subplots(1, len(FAMILY_META), figsize=(15, 4), sharey=False)
    fig.suptitle("Val metric training curves by L (mean ± 1σ over seeds)", fontsize=13)

    cmap = plt.colormaps["viridis"]

    for ax, (family, meta) in zip(axes, FAMILY_META.items()):
        col = meta["col"]
        family_runs = runs.get(family, {})
        l_values = sorted(family_runs)
        n = len(l_values)

        for i, l_val in enumerate(l_values):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is None:
                ax.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i*0.07),
                            xycoords="axes fraction", fontsize=8, color="red")
                continue
            mean_curve, std_curve = agg
            color = cmap(i / max(n - 1, 1))
            ax.plot(mean_curve.index, mean_curve.values, color=color,
                    linewidth=1.5, label=f"L={l_val}")
            ax.fill_between(mean_curve.index,
                            mean_curve.values - std_curve.values,
                            mean_curve.values + std_curve.values,
                            color=color, alpha=0.2)

        ax.set_title(family)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(meta["label"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_nb_diagnostics(runs, figures_dir: Path):
    """Two-panel figure: NB val_nll and theta_mean trajectories per L (mean ± 1σ)."""
    nb_runs = runs.get("nb", {})
    l_values = sorted(nb_runs)
    n = len(l_values)
    cmap = plt.colormaps["viridis"]

    fig, (ax_nll, ax_theta) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NB-RBM: NLL and θ trajectories by L (mean ± 1σ over seeds)", fontsize=13)

    for i, l_val in enumerate(l_values):
        csvs = nb_runs[l_val]
        color = cmap(i / max(n - 1, 1))
        label = f"L={l_val}"

        nll_agg = aggregate_curves(csvs, "val_nll")
        theta_agg = aggregate_curves(csvs, "theta_mean")

        if nll_agg is not None:
            nll_mean, nll_std = nll_agg
            ax_nll.plot(nll_mean.index, nll_mean.values, color=color, linewidth=1.5, label=label)
            ax_nll.fill_between(nll_mean.index,
                                nll_mean.values - nll_std.values,
                                nll_mean.values + nll_std.values,
                                color=color, alpha=0.2)
        else:
            ax_nll.annotate(f"L={l_val}: diverged", xy=(0.05, 0.05 + i * 0.07),
                            xycoords="axes fraction", fontsize=8, color="red")

        if theta_agg is not None:
            theta_mean, theta_std = theta_agg
            ax_theta.plot(theta_mean.index, theta_mean.values, color=color, linewidth=1.5, label=label)
            ax_theta.fill_between(theta_mean.index,
                                  theta_mean.values - theta_std.values,
                                  theta_mean.values + theta_std.values,
                                  color=color, alpha=0.2)

    ax_nll.set_title("Val NLL (↓)")
    ax_nll.set_xlabel("Epoch")
    ax_nll.set_ylabel("NLL")
    ax_nll.legend(fontsize=8)
    ax_nll.grid(True, alpha=0.3)

    ax_theta.set_title("θ mean (dispersion)")
    ax_theta.set_xlabel("Epoch")
    ax_theta.set_ylabel("θ")
    ax_theta.legend(fontsize=8)
    ax_theta.grid(True, alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "sweep_nb_diagnostics.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def print_improvement_table(runs):
    """Print relative val metric improvement per L step for each family."""
    print("\n── Relative improvement per L step ──")
    print(f"{'family':<22} {'L→L+1':<10} {'val metric':<18} {'Δ abs':<10} {'Δ %'}")
    print("-" * 75)

    for family, meta in FAMILY_META.items():
        col = meta["col"]
        better = meta["better"]
        family_runs = runs.get(family, {})

        pairs: list[tuple[int, float, float]] = []
        for l_val in sorted(family_runs):
            agg = aggregate_curves(family_runs[l_val], col)
            if agg is not None:
                mean_curve, std_curve = agg
                pairs.append((l_val, mean_curve.iloc[-1], std_curve.iloc[-1]))

        for i in range(len(pairs) - 1):
            l_a, v_a, s_a = pairs[i]
            l_b, v_b, s_b = pairs[i + 1]
            delta = v_b - v_a
            rel   = delta / abs(v_a) * 100
            marker = ""
            if better == "higher" and delta > 0:
                marker = "↑"
            elif better == "lower" and delta < 0:
                marker = "↓"
            print(f"{family:<22} {l_a}→{l_b:<8} {v_a:.4f}±{s_a:.4f}  {delta:+.4f}   {rel:+.2f}% {marker}")
        print()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(RESULTS_DIR)

    print("Discovered runs:")
    for family, ls in sorted(runs.items()):
        details = ", ".join(f"L{l}({len(ls[l])} seeds)" for l in sorted(ls.keys()))
        print(f"  {family}: {details}")

    print_improvement_table(runs)
    plot_final_metric(runs, FIGURES_DIR)
    plot_training_curves(runs, FIGURES_DIR)
    plot_nb_diagnostics(runs, FIGURES_DIR)


if __name__ == "__main__":
    main()
