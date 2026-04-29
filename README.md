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
| Analysis findings (pre-git)  | CHANGELOG.md    |

---

## Project Structure

```
rbm-plankton/
├── ARCHITECTURE.md         # Technical spec (models, pipeline, monitoring)
├── DECISION_LOG.md         # ADR log — why each choice was made
├── ROADMAP.md              # What is in progress, next, and blocked
├── CHANGELOG.md            # Analysis findings pre-git
├── README.md               # This file
│
├── data/
│   └── raw/
│       └── TimeSeries_countsuL_clean.csv
│
├── src/
│   ├── config.py           # All hyperparameters and paths
│   ├── main.py             # Training entry point
│   ├── data.py             # Data loading and preprocessing
│   ├── utils.py            # Device, save/load weights
│   ├── visualization.py    # Plots and CSV export
│   └── models/
│       ├── base_rbm.py
│       ├── bernoulli_rbm.py
│       └── nb_rbm.py
│
├── results/                # One subdir per run: {model}_L{n_hidden}/
├── requirements.txt
└── .gitignore
```

---

## Setup

### Using uv (recommended)

```bash
sudo apt install uv
uv venv
source .venv/bin/activate
uv sync
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Configure `src/config.py` (model, hyperparameters, paths), then:

```bash
python src/main.py
```

Results are written to `results/{VISIBLE_MODEL}_L{N_HIDDEN}/`.

**Key config switches:**

| Parameter       | Options                | Effect                                  |
| --------------- | ---------------------- | --------------------------------------- |
| `VISIBLE_MODEL` | `"bernoulli"` / `"nb"` | Bernoulli-Bernoulli or NB-Bernoulli RBM |
| `N_HIDDEN`      | integer                | Number of hidden units                  |
| `COUNT_SCALE`   | `1000` (NB only)       | Scales organisms/μL to count scale      |

---
