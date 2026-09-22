"""
config.py - Experiment configuration for RBM-plankton training.
=================================================================
Tweak any parameter here before running `train.py`.
See `models/_constants.py` for model-internal numerical stability guards
(parameter clamps, epsilon floors, saturation thresholds) that should
not normally need adjustment.

Usage:
    from config import ...
"""

# noqa: F401 — CHRONO is a valid SPLIT value, DATA_PATH is re-exported to train.py
from models.paths import CHRONO, DATA_PATH, RUNS_ROOT, SHUFFLED  # noqa: F401

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------
#
# Both come from models/paths.py, which owns every location in the working tree.
# Set RBM_PLANKTON_ROOT to run against a different tree.
#
#   DATA_PATH   raw CSV: daily plankton counts, 83 taxa, organisms/uL
#               (gitignored — user must provide)
#   OUT_ROOT    training runs: {OUT_ROOT}/{family}_L{L}{split}/seed_{k}/

OUT_ROOT = RUNS_ROOT


# ---------------------------------------------------------------------------
# Single-run mode (default)
# ---------------------------------------------------------------------------

# When True: train ONE model (SINGLE_RUN_FAMILY, SINGLE_RUN_L, seed 0).
# No ProcessPoolExecutor, no multi-seed — just runs directly.
# When False: use L_VALUES below for a multi-family/L/seed sweep.
SINGLE_RUN = True

SINGLE_RUN_FAMILY = "nb_sigmoid"
SINGLE_RUN_L      = 6
SINGLE_RUN_SEED   = 0


# ---------------------------------------------------------------------------
# Sweep configuration  (used only when SINGLE_RUN=False)
# ---------------------------------------------------------------------------

# Model families and which L (hidden unit count) values to train.
# Set the list to [] to skip a family entirely.
L_VALUES = {
    "nb":               [],
    "zinb":             [],
    "nb_sigmoid":       [],
    "nb_softmax":       [],
    "bernoulli_median": [],
    "bernoulli_zero":   [],
    "zinb_sigmoid":     [4, 5, 6, 7, 8],
    "zinb_softmax":     [4, 5, 6, 7, 8],
}

# Number of random seeds per (family, L) combination.
N_SEEDS = 10

# Parallel workers for ProcessPoolExecutor.
# Set to 1 to disable parallelism (useful for debugging).
MAX_WORKERS = 10


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

# Total training epochs.
EPOCHS = 1000

# Initial learning rate for RMSprop.
LR = 0.01

# Per-epoch multiplicative LR decay: lr *= LR_DECAY.
# 0.998^1000 = 0.135 — final lr is ~13.5% of initial.
LR_DECAY = 0.998

# Gibbs sampling steps per batch (CD-k / PCD-k).
CD_STEPS = 1

# Initial and final batch sizes for the quadratic batch-size schedule.
# Batch size at epoch e = int(BATCH_I + (BATCH_F - BATCH_I) * ((e-1)/(E-1))^2).
# Small early batches give noisy gradients that help escape bad local minima.
# Larger final batches give stable gradients at convergence.
BATCH_I = 10
BATCH_F = 256

# Number of random mini-batches per epoch (drawn with replacement).
# Total sample-steps per epoch = N_BATCHES * mean(batch_size).
N_BATCHES = 20

# L1 regularisation coefficient on W (gamma * lr * sign(W) added to dW).
GAMMA = 1e-4

# RMSprop momentum (EMA coefficient for squared gradients).
BETA = 0.9

# RMSprop denominator floor: update = lr * g / sqrt(s + EPSILON).
EPSILON = 1e-4

# Fraction of the main LR used for theta (dispersion) and pi (zero-inflation)
# parameter updates via autograd.
LR_PARAM_MULTIPLIER = 0.1


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------

# Fraction of clean rows reserved for validation.
# Remaining (1 - VAL_FRAC) used for training.
VAL_FRAC = 0.15

# Split strategy — SHUFFLED or CHRONO.
#   SHUFFLED  shuffle before splitting: random, i.i.d. validation set
#   CHRONO    chronological split: first 85% train, last 15% val (a future window)
# This single value decides both how the data is split and the run-directory
# suffix (io.split_suffix), so the two can no longer disagree.
SPLIT = SHUFFLED


# ---------------------------------------------------------------------------
# Data scaling
# ---------------------------------------------------------------------------

# Multiplier applied to raw organisms/uL values before training.
# 1000 brings data from [0, 0.444] organisms/uL to approximate integer scale,
# which prevents lgamma gradient collapse in NB visible units.
# Applied uniformly across all model families (Bernoulli and NB/ZINB).
COUNT_SCALE = 1000


# ---------------------------------------------------------------------------
# NB / ZINB family settings
# ---------------------------------------------------------------------------

# Initial log-dispersion value: theta = exp(THETA_INIT_LOG).
# 0.0 gives theta = 1.0 (Poisson-like dispersion at init).
THETA_INIT_LOG = 0.0

# If True: use Persistent Contrastive Divergence (PCD-k).
# PCD maintains fantasy particles across batches, improving mixing for
# NB/ZINB models with multimodal hidden distributions.
USE_PCD = True

# Number of persistent fantasy particles.
# Must be >= BATCH_F so that each batch draws unique particles.
N_PCD_CHAINS = 500


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

# Evaluate validation metrics every EVAL_EVERY epochs.
EVAL_EVERY = 10
