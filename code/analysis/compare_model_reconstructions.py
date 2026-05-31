"""compare_model_reconstructions.py

Compare one-step reconstructions from all trained RBM families for a given
hidden size L and seed.

The script loads one random clean sample from the plankton time series CSV,
runs a single reconstruction step for each requested model, and saves a plot
with the original profile, reconstructed profiles, residuals, a top-3 view,
and a test-set summary bar plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.models import io as data_io
import use_trained_rbm as trained_loader

MODEL_FAMILIES = [
    ("bernoulli_median", "bernoulli_median"),
    ("bernoulli_zero", "bernoulli_zero"),
    ("nb", "nb"),
    ("nb_relu", "nb_relu"),
    ("nb_sigmoid", "nb_sigmoid"),
    ("nb_softmax", "nb_softmax"),
    ("zinb", "zinb"),
    ("zinb_relu", "zinb_relu"),
    ("zinb_sigmoid", "zinb_sigmoid"),
]

OVERALL_RANK_MIX = [0.52, 0.32, 0.10]
OVERALL_RANK_LABELS = ["worst pseudo-LL", "middle pseudo-LL", "best pseudo-LL"]

LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
    (0, (5, 2, 1, 2)),
    (0, (1, 1)),
    (0, (2, 2, 6, 2)),
]


def load_clean_dataframe(data_path: Path):
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    taxa_cols = [col for col in df.columns if col != "date"]

    nonzero_mask = df[taxa_cols].sum(axis=1) > 0
    df = df[nonzero_mask].copy()

    nan_mask = df[taxa_cols].isna().any(axis=1)
    df = df[~nan_mask].copy().reset_index(drop=True)

    return df, taxa_cols


def compute_thresholds(df: pd.DataFrame, taxa_cols: list[str], scale: float = 1000.0):
    values = df[taxa_cols].to_numpy(dtype=np.float32) * scale
    return np.median(values, axis=0)


def normalize_profile(values, eps: float = 1e-6):
    array = np.asarray(values, dtype=np.float64)
    array = np.clip(array, 0.0, None)
    total = array.sum()
    if total <= eps:
        return np.zeros_like(array)
    return array / total


def load_model_run(model_name: str, hidden_units: int, seed: int, device: str):
    weights_path = (
        ROOT
        / "training_runs"
        / f"{model_name}_L{hidden_units}_shuffled"
        / f"seed_{seed}"
        / "weights.npz"
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    npz = trained_loader.load_weights_npz(weights_path)
    model, visible_model = trained_loader.instantiate_model_from_weights(
        npz,
        device_str=device,
        override_class=None,
    )
    return model, visible_model, weights_path


def reconstruct_once(model: torch.nn.Module, visible: np.ndarray, device: torch.device):
    tensor = torch.tensor(visible, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model.reconstruct(tensor.unsqueeze(0)).squeeze(0)
    return recon.detach().cpu().numpy()


def prepare_model_input(visible_model: str, raw_profile: np.ndarray, thresholds: np.ndarray):
    if visible_model.startswith("bernoulli"):
        return (raw_profile > thresholds).astype(np.float32)
    return raw_profile.astype(np.float32)


def supports_pseudo_log_likelihood(visible_model: str):
    return not visible_model.startswith("bernoulli")


def reconstruct_and_distance(
    model: torch.nn.Module,
    visible_model: str,
    raw_profile: np.ndarray,
    thresholds: np.ndarray,
    device: torch.device,
):
    model_input = prepare_model_input(visible_model, raw_profile, thresholds)
    recon = reconstruct_once(model, model_input, device)
    distance = float(np.linalg.norm(recon - raw_profile))
    return recon, distance


def pseudo_log_likelihood(
    model: torch.nn.Module,
    visible_model: str,
    raw_profile: np.ndarray,
    thresholds: np.ndarray,
    device: torch.device,
):
    if not supports_pseudo_log_likelihood(visible_model):
        return None
    model_input = prepare_model_input(visible_model, raw_profile, thresholds)
    tensor = torch.tensor(model_input, dtype=torch.float32, device=device).unsqueeze(0)
    if hasattr(model, "pll"):
        return -float(model.pll(tensor))
    if hasattr(model, "nll"):
        return -float(model.nll(tensor))
    raise AttributeError(f"Model {type(model).__name__} does not expose pll() or nll()")


def evaluate_test_set_metrics(
    model: torch.nn.Module,
    visible_model: str,
    test_df: pd.DataFrame,
    taxa_cols: list[str],
    thresholds: np.ndarray,
    device: torch.device,
):
    distances = []
    pseudo_lls = []
    for _, row in test_df.iterrows():
        raw_profile = row[taxa_cols].to_numpy(dtype=np.float32) * 1000.0
        _, distance = reconstruct_and_distance(
            model,
            visible_model,
            raw_profile,
            thresholds,
            device,
        )
        distances.append(distance)
        pseudo_ll = pseudo_log_likelihood(
            model,
            visible_model,
            raw_profile,
            thresholds,
            device,
        )
        if pseudo_ll is not None:
            pseudo_lls.append(pseudo_ll)

    avg_distance = float(np.mean(distances))
    avg_pseudo_ll = float(np.mean(pseudo_lls)) if pseudo_lls else None
    return avg_distance, avg_pseudo_ll


def sem(values: list[float]):
    if len(values) <= 1:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.std(ddof=1) / np.sqrt(len(array)))


def shade_color(color, mix: float):
    base = np.asarray(mcolors.to_rgb(color), dtype=np.float64)
    white = np.ones(3, dtype=np.float64)
    return tuple((1.0 - mix) * base + mix * white)


def family_base_color(family_label: str):
    family_order = [label for _, label in MODEL_FAMILIES]
    family_idx = family_order.index(family_label)
    return plt.get_cmap("tab10")(family_idx % 10)


def evaluate_overall_results(
    taxa_cols: list[str],
    thresholds: np.ndarray,
    device: torch.device,
    test_df: pd.DataFrame,
    hidden_units_range: range,
    seeds: range,
):
    overall = {}
    for model_name, label in MODEL_FAMILIES:
        per_l = {}
        best_run = None
        supports_pll = None
        for hidden_units in hidden_units_range:
            seed_distances = []
            seed_pseudo_lls = []
            for seed in seeds:
                try:
                    model, visible_model, _ = load_model_run(model_name, hidden_units, seed, device.type)
                except FileNotFoundError:
                    print(f"[Overall] skip {label} L={hidden_units} seed={seed}: missing weights")
                    continue
                avg_distance, avg_pseudo_ll = evaluate_test_set_metrics(
                    model,
                    visible_model,
                    test_df,
                    taxa_cols,
                    thresholds,
                    device,
                )
                seed_distances.append(avg_distance)
                if avg_pseudo_ll is not None:
                    seed_pseudo_lls.append(avg_pseudo_ll)
                    if best_run is None or avg_pseudo_ll > best_run["pll_mean"]:
                        best_run = {
                            "model_name": model_name,
                            "label": label,
                            "L": hidden_units,
                            "seed": seed,
                            "distance_mean": avg_distance,
                            "pll_mean": avg_pseudo_ll,
                        }
                        supports_pll = True
                elif supports_pll is None:
                    supports_pll = False
                elif supports_pll is False:
                    pass
                else:
                    pass
                print(
                    f"[Overall] {label} L={hidden_units} seed={seed}: "
                    f"avg_distance={avg_distance:.4f}, avg_pseudo_ll={avg_pseudo_ll if avg_pseudo_ll is not None else 'N/A'}"
                )

            if not seed_distances:
                continue

            per_l[hidden_units] = {
                "distance_mean": float(np.mean(seed_distances)),
                "distance_sem": sem(seed_distances),
                "pll_mean": float(np.mean(seed_pseudo_lls)) if seed_pseudo_lls else None,
                "pll_sem": sem(seed_pseudo_lls) if seed_pseudo_lls else None,
            }
        if not per_l:
            print(f"[Overall] {label}: no weights found in L={hidden_units_range.start}..{hidden_units_range.stop - 1}")
            continue

        if any(metrics["pll_mean"] is not None for metrics in per_l.values()):
            ranked_ls = sorted(per_l.items(), key=lambda item: item[1]["pll_mean"], reverse=True)
            selected_ls = ranked_ls[:3]
            top_l = sorted(
                [
                    {
                        "L": hidden_units,
                        "distance_mean": metrics["distance_mean"],
                        "distance_sem": metrics["distance_sem"],
                        "pll_mean": metrics["pll_mean"],
                        "pll_sem": metrics["pll_sem"],
                    }
                    for hidden_units, metrics in selected_ls
                ],
                key=lambda entry: entry["pll_mean"],
            )
        else:
            ranked_ls = sorted(per_l.items(), key=lambda item: item[1]["distance_mean"], reverse=True)
            selected_ls = ranked_ls[:3]
            top_l = sorted(
                [
                    {
                        "L": hidden_units,
                        "distance_mean": metrics["distance_mean"],
                        "distance_sem": metrics["distance_sem"],
                        "pll_mean": None,
                        "pll_sem": None,
                    }
                    for hidden_units, metrics in selected_ls
                ],
                key=lambda entry: entry["distance_mean"],
                reverse=True,
            )
        overall[label] = {
            "per_l": per_l,
            "top_l": top_l,
            "family_sort_distance": max(entry["distance_mean"] for entry in top_l),
            "best_run": best_run,
        }
        top_l_parts = []
        for entry in overall[label]["top_l"]:
            if entry["pll_mean"] is None:
                pll_text = "N/A"
            else:
                pll_text = f"{entry['pll_mean']:.4f} ± {entry['pll_sem']:.4f}"
            top_l_parts.append(
                f"L={entry['L']} (distance={entry['distance_mean']:.4f} ± {entry['distance_sem']:.4f}, pseudo_ll={pll_text})"
            )
        print(f"[Overall] {label}: top_Ls=" + ", ".join(top_l_parts))
        if best_run is not None:
            print(
                f"[Overall] {label}: best_seed=L{best_run['L']}, seed={best_run['seed']}, "
                f"distance={best_run['distance_mean']:.4f}, pseudo_ll={best_run['pll_mean']:.4f}"
            )

    return overall


def write_best_seed_report(overall_results: dict, out_path: Path):
    lines = ["model\tbest_L\tbest_seed\tavg_distance\tavg_pseudo_ll"]
    for label, result in overall_results.items():
        best_run = result.get("best_run")
        if best_run is None:
            lines.append(f"{label}\tN/A\tN/A\tN/A\tN/A")
            continue
        lines.append(
            f"{label}\t{best_run['L']}\t{best_run['seed']}\t"
            f"{best_run['distance_mean']:.6f}\t{best_run['pll_mean']:.6f}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Save] {out_path}")


def evaluate_average_residual_profile(
    model: torch.nn.Module,
    visible_model: str,
    test_df: pd.DataFrame,
    taxa_cols: list[str],
    thresholds: np.ndarray,
    device: torch.device,
    raw: bool,
):
    residuals = []
    for _, row in test_df.iterrows():
        raw_profile = row[taxa_cols].to_numpy(dtype=np.float32) * 1000.0
        original_profile = raw_profile if raw else normalize_profile(raw_profile)
        recon_raw, _ = reconstruct_and_distance(
            model,
            visible_model,
            raw_profile,
            thresholds,
            device,
        )
        recon_profile = recon_raw if raw else normalize_profile(recon_raw)
        residuals.append(recon_profile - original_profile)

    return np.mean(np.asarray(residuals, dtype=np.float64), axis=0)


def plot_overall_summary(
    overall_results: dict,
    out_path: Path,
    residual_profile: np.ndarray,
    taxa_cols: list[str],
    residual_title: str,
    residual_color,
):
    family_order = sorted(
        [label for _, label in MODEL_FAMILIES if label in overall_results],
        key=lambda label: overall_results[label]["family_sort_distance"],
        reverse=True,
    )
    rank_mix = OVERALL_RANK_MIX
    rank_labels = OVERALL_RANK_LABELS

    entries = []
    for family_idx, label in enumerate(family_order):
        base_color = plt.get_cmap("tab10")(family_idx % 10)
        for rank_idx, entry in enumerate(overall_results[label]["top_l"]):
            entries.append(
                {
                    "family": label,
                    "L": entry["L"],
                    "distance_mean": entry["distance_mean"],
                    "distance_sem": entry["distance_sem"],
                    "pll_mean": entry["pll_mean"],
                    "pll_sem": entry["pll_sem"],
                    "color": shade_color(base_color, rank_mix[rank_idx]),
                    "rank": rank_idx,
                }
            )

    if not entries:
        raise RuntimeError("No overall results available to plot")

    x_pos = np.arange(len(entries))
    x_labels = [f"{entry['family']}\n(L = {entry['L']})" for entry in entries]
    distances = [entry["distance_mean"] for entry in entries]
    distance_sems = [entry["distance_sem"] for entry in entries]
    plls = [entry["pll_mean"] for entry in entries]
    pll_sems = [entry["pll_sem"] for entry in entries]
    colors = [entry["color"] for entry in entries]

    fig = plt.figure(figsize=(max(18, len(entries) * 0.85), 11.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35])
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_resid = fig.add_subplot(gs[1, 0])
    fig.suptitle("Overall top-3 hidden sizes per model family", fontsize=14)

    ax_dist.bar(
        x_pos,
        distances,
        yerr=distance_sems,
        color=colors,
        width=0.75,
        capsize=4,
        alpha=0.95,
    )
    ax_dist.set_xticks(x_pos)
    ax_dist.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_dist.set_ylabel("Average Euclidean distance")
    ax_dist.set_xlabel("Model family and hidden units (L = value)")
    ax_dist.grid(axis="y", alpha=0.3)

    ax_pll = ax_dist.twinx()
    pll_positions = []
    pll_values = []
    for xpos, pll_value, pll_sem, color in zip(x_pos, plls, pll_sems, colors):
        if pll_value is None:
            continue
        pll_positions.append(xpos)
        pll_values.append(pll_value)
        ax_pll.errorbar(
            xpos,
            pll_value,
            yerr=pll_sem,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=5,
        )
    if pll_positions:
        ax_pll.plot(pll_positions, pll_values, color="0.35", linewidth=1.0, alpha=0.7, zorder=4)
    ax_pll.set_ylabel("Average pseudo log-likelihood\n(higher is better)")

    rank_handles = [
        Line2D([0], [0], color=shade_color("#666666", mix), lw=8, label=label)
        for mix, label in zip(rank_mix, rank_labels)
    ]
    ax_dist.legend(handles=rank_handles, loc="upper left", fontsize=8, frameon=False)

    resid_color = "black"
    ax_resid.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.7)
    ax_resid.plot(np.arange(len(taxa_cols)), residual_profile, color=residual_color, linewidth=1.6)
    ax_resid.set_xticks(np.arange(len(taxa_cols)))
    ax_resid.set_xticklabels(taxa_cols, rotation=90, fontsize=6)
    ax_resid.set_ylabel("Avg residual")
    ax_resid.set_title(residual_title)
    ax_resid.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[Save] {out_path}")


def main():
    epilog = (
        "Examples:\n"
        "  python scripts/compare_model_reconstructions.py --Ln 6 --seed 0\n\n"
        "  python scripts/compare_model_reconstructions.py --Ln 9 --seed 2 --sample-seed 123\n\n"
        "  python scripts/compare_model_reconstructions.py --Ln 6 --seed 0 --raw\n\n"

        "  python scripts/compare_model_reconstructions.py --overall --test 150\n\n"

        "  python scripts/compare_model_reconstructions.py --Ln 6 --seed 0 --plot-out results/reconstruction_plots/custom.png\n"
    )

    parser = argparse.ArgumentParser(
        description="Compare one-step reconstructions from multiple RBM families.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--Ln", type=int, default=None, help="Number of hidden units L used during training")
    parser.add_argument("--seed", type=int, default=None, help="Seed index used during training")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed used to choose the random sample")
    parser.add_argument("--data-path", type=str, default=data_io.DATA_PATH, help="Path to the plankton CSV file")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Torch device")
    parser.add_argument("--raw", action="store_true", help="Plot raw abundances instead of normalized abundances")
    parser.add_argument("--test", type=int, default=150, help="Number of last clean entries to use for average-distance evaluation")
    parser.add_argument("--overall", action="store_true", help="Compare models across L=3..8 and seeds 0..9, then plot the best L per model")
    parser.add_argument("--plot-out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.overall and (args.Ln is None or args.seed is None):
        parser.error("--Ln and --seed are required unless --overall is set")

    df, taxa_cols = load_clean_dataframe(Path(args.data_path))
    thresholds = compute_thresholds(df, taxa_cols)

    n_test = min(max(args.test, 1), len(df))
    test_df = df.tail(n_test).copy().reset_index(drop=True)

    if args.overall:
        device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
        hidden_units_range = range(3, 9)
        seed_range = range(10)
        overall_results = evaluate_overall_results(
            taxa_cols,
            thresholds,
            device,
            test_df,
            hidden_units_range,
            seed_range,
        )

        if args.plot_out is None:
            out_path = ROOT / "results" / "reconstruction_plots" / "compare_generation_overall.png"
        else:
            out_path = Path(args.plot_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.name == "compare_generation_overall.png":
                pass

        report_path = out_path.with_name("compare_generation_overall_best_seed.txt")
        write_best_seed_report(overall_results, report_path)

        best_candidates = [
            (label, result["best_run"])
            for label, result in overall_results.items()
            if result.get("best_run") is not None
        ]
        if not best_candidates:
            raise RuntimeError("No non-Bernoulli best run available for residual plot")
        best_family_label, best_run = max(best_candidates, key=lambda item: item[1]["pll_mean"])
        best_model, best_visible_model, _ = load_model_run(
            best_run["model_name"],
            best_run["L"],
            best_run["seed"],
            args.device,
        )
        best_residual_profile = evaluate_average_residual_profile(
            best_model,
            best_visible_model,
            test_df,
            taxa_cols,
            thresholds,
            device,
            args.raw,
        )

        family_order = sorted(
            [label for _, label in MODEL_FAMILIES if label in overall_results],
            key=lambda label: overall_results[label]["family_sort_distance"],
            reverse=True,
        )
        base_color = plt.get_cmap("tab10")(family_order.index(best_family_label) % 10)
        family_top_l = overall_results[best_family_label]["top_l"]
        residual_color = None
        for rank, entry in enumerate(family_top_l):
            if entry["L"] == best_run["L"]:
                residual_color = shade_color(base_color, OVERALL_RANK_MIX[rank])
                break
        if residual_color is None:
            residual_color = base_color

        residual_title = (
            f"Average residuals for best run: {best_family_label} | "
            f"L={best_run['L']} seed={best_run['seed']}"
        )

        plot_overall_summary(
            overall_results,
            out_path,
            best_residual_profile,
            taxa_cols,
            residual_title,
            residual_color,
        )
        return

    rng = np.random.default_rng(args.sample_seed)
    sample_idx = int(rng.integers(0, len(df)))
    sample_row = df.iloc[sample_idx]
    sample_date = sample_row["date"]

    raw_profile = sample_row[taxa_cols].to_numpy(dtype=np.float32) * 1000.0
    original_profile = normalize_profile(raw_profile)
    original_profile = raw_profile if args.raw else normalize_profile(raw_profile)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print(f"[Sample] index={sample_idx} date={sample_date.date()}")
    print(f"[Sample] taxa={len(taxa_cols)}  sum={raw_profile.sum():.4f}  device={device}")

    x_axis = np.arange(len(taxa_cols))
    models = []
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(MODEL_FAMILIES)))

    for model_name, label in MODEL_FAMILIES:
        model, visible_model, weights_path = load_model_run(model_name, args.Ln, args.seed, args.device)
        models.append(
            {
                "model_name": model_name,
                "label": label,
                "model": model,
                "visible_model": visible_model,
                "weights_path": weights_path,
            }
        )
        print(f"[Load] {label}: {weights_path}")

    sample_reconstructions = []
    sample_residuals = []
    sample_raw_distances = []
    labels = []

    for info in models:
        recon_raw, raw_distance = reconstruct_and_distance(
            info["model"],
            info["visible_model"],
            raw_profile,
            thresholds,
            device,
        )
        recon_plot = recon_raw if args.raw else normalize_profile(recon_raw)
        residual = recon_plot - original_profile

        labels.append(info["label"])
        sample_reconstructions.append(recon_plot)
        sample_residuals.append(residual)
        sample_raw_distances.append(raw_distance)

    avg_distances = []
    avg_plls = []
    pll_sems = []
    for info in models:
        total_distance = 0.0
        total_pll = 0.0
        pll_scores = []
        for _, row in test_df.iterrows():
            test_raw_profile = row[taxa_cols].to_numpy(dtype=np.float32) * 1000.0
            _, distance = reconstruct_and_distance(
                info["model"],
                info["visible_model"],
                test_raw_profile,
                thresholds,
                device,
            )
            pll_score = pseudo_log_likelihood(
                info["model"],
                info["visible_model"],
                test_raw_profile,
                thresholds,
                device,
            )
            total_distance += distance
            if pll_score is not None:
                total_pll += pll_score
                pll_scores.append(pll_score)
        avg_distance = total_distance / n_test
        avg_pll = (total_pll / len(pll_scores)) if pll_scores else None
        pll_std = float(np.std(pll_scores, ddof=1)) if n_test > 1 else 0.0
        pll_sem = pll_std / np.sqrt(n_test) if n_test > 1 else 0.0
        avg_distances.append(avg_distance)
        avg_plls.append(avg_pll)
        pll_sems.append(pll_sem)
        print(f"[Test] {info['label']}: avg_distance={avg_distance:.4f} over {n_test} entries")
        pll_text = f"{avg_pll:.4f}" if avg_pll is not None else "N/A"
        print(f"[Test] {info['label']}: avg_pseudo_ll={pll_text} ± {pll_sem:.4f} over {n_test} entries")

    fig, axes = plt.subplots(4, 1, figsize=(20, 19))
    fig.suptitle(f"One-step reconstructions at L={args.Ln}, seed={args.seed} | sample {sample_idx} ({sample_date.date()})", fontsize=14)

    # Top: original + all reconstructions
    axes[0].plot(
        x_axis,
        original_profile,
        color="black",
        linewidth=2.5,
        label="original",
    )
    for idx, (label, color, linestyle, recon) in enumerate(zip(labels, colors, LINE_STYLES, sample_reconstructions)):
        axes[0].plot(
            x_axis,
            recon,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    axes[0].set_ylabel("Raw abundance" if args.raw else "Relative abundance")
    axes[0].set_title("Original and reconstructed species distributions")
    axes[0].grid(alpha=0.3)

    # Middle: residuals for all models
    for idx, (label, color, linestyle, residual) in enumerate(zip(labels, colors, LINE_STYLES, sample_residuals)):
        axes[1].plot(
            x_axis,
            residual,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("Residual (reconstructed - original)")
    axes[1].set_title("Residuals by model")
    axes[1].grid(alpha=0.3)

    # Determine best 3 models by Euclidean distance (lower is better)
    distances = sample_raw_distances
    num_best = min(3, len(distances))
    best_idxs = list(np.argsort(distances)[:num_best])

    # Bottom: original + top-3 reconstructions (use same colors/linestyles)
    # Capture plot handles so we can create a legend with distances
    bottom_handles = []
    bottom_labels = []
    orig_handle = axes[2].plot(
        x_axis,
        original_profile,
        color="black",
        linewidth=2.5,
        label="original",
    )[0]
    bottom_handles.append(orig_handle)
    bottom_labels.append("original")

    for idx in best_idxs:
        h = axes[2].plot(
            x_axis,
            sample_reconstructions[idx],
            color=colors[idx],
            linestyle=LINE_STYLES[idx],
            linewidth=1.6,
            label=labels[idx],
        )[0]
        bottom_handles.append(h)
        bottom_labels.append(labels[idx])
    axes[2].set_ylabel("Raw abundance" if args.raw else "Relative abundance")
    axes[2].set_title(f"Top-{num_best} reconstructions (best by Euclidean distance)")
    axes[2].grid(alpha=0.3)

    # Add legend for the bottom subplot showing the best models with distances
    distance_labels = [
        f"{lbl} ({distances[idx]:.1f})" if lbl != "original" else "original"
        for lbl, idx in zip(bottom_labels, [-1] + best_idxs)
    ]
    # The first entry corresponds to original; distances list uses indices for models
    # Build appropriate handles and labels (skip distance for original)
    axes[2].legend(bottom_handles, distance_labels, loc="upper right", fontsize=8, frameon=False)

    axes[2].set_xticks(x_axis)
    axes[2].set_xticklabels(taxa_cols, rotation=90, fontsize=6)

    order = np.argsort(avg_distances)[::-1]
    sorted_labels = [labels[idx] for idx in order]
    sorted_distances = [avg_distances[idx] for idx in order]
    sorted_colors = [colors[idx] for idx in order]

    axes[3].bar(np.arange(len(sorted_labels)), sorted_distances, color=sorted_colors, width=0.75)
    axes[3].set_xticks(np.arange(len(sorted_labels)))
    axes[3].set_xticklabels(sorted_labels, rotation=45, ha="right")
    axes[3].set_ylabel(f"Average Euclidean distance\n(last {n_test} entries)")
    axes[3].set_title("Test-set average distance by model")
    axes[3].grid(axis="y", alpha=0.3)

    ax_pll = axes[3].twinx()
    sorted_plls = [avg_plls[idx] for idx in order]
    sorted_pll_sems = [pll_sems[idx] for idx in order]
    x_pos = np.arange(len(sorted_labels))
    for xpos, pll_value, pll_sem, color in zip(x_pos, sorted_plls, sorted_pll_sems, sorted_colors):
        if pll_value is None:
            continue
        ax_pll.errorbar(
            xpos,
            pll_value,
            yerr=pll_sem,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=5,
        )
    sorted_pll_positions = [x for x, value in zip(x_pos, sorted_plls) if value is not None]
    sorted_pll_values = [value for value in sorted_plls if value is not None]
    if sorted_pll_values:
        ax_pll.plot(sorted_pll_positions, sorted_pll_values, color="0.35", linewidth=1.0, alpha=0.7, zorder=4)
    ax_pll.set_ylabel("Average pseudo log-likelihood\n(higher is better)")

    fig.legend(loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    if args.plot_out is None:
        out_dir = ROOT / "results" / "reconstruction_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_raw" if args.raw else ""
        out_path = out_dir / f"compare_generation_L{args.Ln}_seed_{args.seed}{suffix}.png"
    else:
        out_path = Path(args.plot_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.raw:
            stem_suffix = "_raw"
            if not out_path.stem.endswith(stem_suffix):
                out_path = out_path.with_name(f"{out_path.stem}{stem_suffix}{out_path.suffix}")

    fig.savefig(out_path, dpi=200)
    print(f"[Save] {out_path}")


if __name__ == "__main__":
    main()
