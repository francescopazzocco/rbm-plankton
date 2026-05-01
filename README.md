# RBM-Plankton

Restricted Boltzmann Machine for unsupervised learning of plankton community
structure from Lake Greifen monitoring data (2019–2024).

## Motivation

Lake Greifen exhibits seasonal plankton dynamics. This project uses an RBM to learn
latent community states from 83 taxa abundance time series and interprets learned
patterns against known ecological cycles.

**Dataset**: Eyring et al. (2025). *Scientific Data* 12:653 — 5 years of high-frequency
phytoplankton and zooplankton observations from Lake Greifen.
DOI: [10.1038/s41597-025-04988-9](https://doi.org/10.1038/s41597-025-04988-9)

---

## Documentation

| Topic                        | Reference       |
| ---------------------------- | --------------- |
| Model architecture and specs | ARCHITECTURE.md |
| Decision log (why)           | DECISION_LOG.md |
| Roadmap and open questions   | ROADMAP.md      |

---

## Project Structure

```
rbm-plankton/
├── ARCHITECTURE.md              # Technical spec (models, pipeline, monitoring)
├── DECISION_LOG.md              # ADR log — why each choice was made
├── ROADMAP.md                   # What is in progress, next, and blocked
├── README.md                    # This file
│
├── data/
│   └── raw/
│       └── TimeSeries_countsuL_clean.csv
│
├── src/
│   ├── main_multiseed.py        # Training pipeline — parallel N-seed runs
│   ├── dataset_analysis.py      # EDA pipeline — dataset structure figures
│   ├── sweep_analysis.py        # L-sweep pipeline — val metric vs L figures
│   ├── hidden_coactivation.py   # Hidden analysis — weight profiles + state timelines
│   ├── hidden_mean_activation.py # Hidden analysis — mean activation per unit
│   ├── hidden_cross_model.py    # Hidden analysis — NB vs BB cross-model comparison
│   └── models/
│       ├── io.py                # File I/O: data loaders + results navigation
│       ├── utils.py             # Shared utilities: device, save/load weights
│       ├── visualization.py     # All plotting functions, organised by pipeline
│       ├── base_rbm.py          # Shared RBM interface and initialisation
│       ├── bernoulli_rbm.py     # BB-RBM: CD-1, pll, hidden_probs
│       └── nb_rbm.py            # NBB-RBM: PCD-1, nll, hidden_probs, θ update
│
├── results/
│   ├── training_runs/           # Canonical model artifacts (weights + CSVs)
│   │   └── {family}_L{n}/seed_{k}/
│   └── figures/
│       ├── dataset_analysis/    # EDA figures
│       ├── sweep/               # L-sweep figures
│       ├── hidden/              # Hidden activation analysis figures and CSVs
│       └── training_runs/       # Per-run training curves (when PLOT_RESULTS=True)
│
├── archive/                     # Superseded runs and scripts (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python -m venv rbm_plankton
source rbm_plankton/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Training

All hyperparameters are defined at the top of `src/main_multiseed.py`.
Edit `L_VALUES`, `N_SEEDS`, `EPOCHS`, `LR`, etc. directly, then run:

```bash
python src/main_multiseed.py
```

Trains all `(family, L, seed)` combinations in parallel. Skips already-completed
runs automatically. Results go to `results/training_runs/{family}_L{n}/seed_{k}/`.

**Model families:**

| Family            | Visible units   | Training   |
| ----------------- | --------------- | ---------- |
| `nb`              | Negative-Binomial | PCD-1    |
| `bernoulli_median`| Bernoulli (median threshold) | CD-1 |
| `bernoulli_zero`  | Bernoulli (zero threshold)   | CD-1 |

### Analysis pipelines

Run from the project root after training:

```bash
python src/sweep_analysis.py          # L-sweep figures → results/figures/sweep/
python src/hidden_coactivation.py     # Weight profiles + state timelines
python src/hidden_mean_activation.py  # Mean activation per unit
python src/hidden_cross_model.py      # NB vs BB-median comparison
```

---

## Status

| Stage | Status |
| ----- | ------ |
| Dataset EDA | Done |
| L-sweep (N=10 seeds, L∈{3,4,5,6,7}, all families) | Done — `results/training_runs/` |
| L selection | Done — **L=6** for all families (LOG-017) |
| Hidden activation analysis | Done — `results/figures/hidden/` |
| Cross-model comparison (NB vs BB-median) | Done |
| NaN test set evaluation | **In progress** — October 2022 structured missingness |

**Key results:**
- L=6 selected: cumulative ~2.5% NLL/PLL gain over L=3; no gain at L=7.
- Both NB-RBM and BB-median independently recover the same two dominant ecological
  axes (summer community, winter community).
- NB uses compositional representation (~30 distinct 6-bit patterns); BB-median uses
  exclusive switching.
- `bernoulli_zero` (zero-threshold binarisation) is near-trivial — flat across all L,
  confirming the median threshold decision (LOG-005).
