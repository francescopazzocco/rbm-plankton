# Analysis scripts

These scripts read saved RBM outputs from `weights/` and write figures/tables to `analysis/results/`.

## `rbm_hidden_stackplot.py`

Plots normalized hidden activations as a stackplot over time.

```bash
conda run -n jupyter python analysis/rbm_hidden_stackplot.py \
  --input-dir weights/NB_RBM/L6_chrono \
  --output analysis/results/nb_rbm_hidden_stackplot.png
```

### Main options
- `--input-dir`: folder containing `rbm_hidden_activations.csv`
- `--output`: output image path
- `--title`: optional figure title

## `hidden_pattern_analysis.py`

Converts hidden probabilities into discrete patterns and saves:
- a frequency histogram
- a timeline plot
- CSV summaries for both

```bash
conda run -n jupyter python analysis/hidden_pattern_analysis.py \
  --input-dir weights/NB_RBM/L6_chrono \
  --mode sigmoid \
  --output-dir analysis/results/patterns_nb
```

### Main options
- `--input-dir`: folder containing `rbm_hidden_activations.csv`
- `--mode`: `sigmoid` or `softmax`
- `--output-dir`: folder for PNG and CSV outputs
- `--title-prefix`: prefix used in plot titles

### Output files
- `pattern_frequency_<mode>.csv`
- `pattern_timeline_<mode>.csv`
- `pattern_histogram_<mode>.png`
- `pattern_timeline_<mode>.png`

## `plot_visible_by_hidden.py`

Plot visible-unit probabilities/expected counts when activating single hidden nodes or full patterns.

```bash
conda run -n jupyter python analysis/plot_visible_by_hidden.py \
  --weights weights/zinb_sigmoid_L7_seed9_best.npz \
  --output-dir analysis/results/visible_by_hidden \
  --mode zinb \
  --patterns-csv analysis/results/nb_sigmoid_patterns/patterns.csv \
  --title-prefix "ZINB RBM"
```

### Main options
- `--weights`: path to `.npz` weights file or CSV (D x L) with taxa index
- `--output-dir`: folder for PNG and CSV outputs (default `analysis/results/visible_by_hidden`)
- `--mode`: `bernoulli` or `zinb` (auto-detected if omitted)
- `--patterns-csv`: optional CSV with column `pattern` containing binary strings
- `--title-prefix`: prefix used in plot titles

Additional options (new):
- `--no-normalize`: plot raw values instead of renormalizing each hidden node to frequency [0,1]
- `--log-y`: display y-axis on logarithmic scale (small values clamped to avoid zeros)

### Y-scale meaning
- Default (no `--no-normalize`): each hidden node row is renormalized to frequencies that sum to 1 across species (y in [0,1]). Useful to compare relative species contribution per hidden unit.
- With `--no-normalize`:
  - `bernoulli` mode → y = probability for each visible species (0..1)
  - `zinb` mode → y = expected count/mean for each visible species (non-negative real)
- With `--log-y`: plot uses logarithmic scale; zeros/small values are clamped to a small epsilon for display. CSV outputs remain raw values (unchanged by plotting flags).

Other notes:
- Marker borders forced black so point edges visible.
- X-axis limits tightened so first/last species sit closer to axis.

### Output files
- `visible_by_hidden_<mode>.png` (per-hidden normalized rows)
- `visible_by_hidden_<mode>.csv` (raw probabilities / expected counts)
- `visible_by_hidden_<mode>_freq.csv` (normalized per-hidden frequencies)
- when `--patterns-csv` provided: `visible_by_patterns_<mode>.png`, `...csv`, `..._freq.csv`

Notes: species names read from `taxa` inside `.npz` or from CSV index; otherwise generic s0..s{D-1}.

## Notes
- The binary label uses left-to-right hidden order: `h0` is the leftmost bit.
- The timeline plot orders patterns by Hamming-distance clustering.
- The timeline points use a rainbow colormap with 75% opacity.
