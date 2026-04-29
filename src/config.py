"""
config.py — Centralized configuration for RBM plankton project
==========================================================
"""


# Training hyperparameters
N_HIDDEN     = 10
EPOCHS       = 500
LR           = 0.01
LR_DECAY     = 0.998
CD_STEPS     = 1
N_BATCHES    = 20
BATCH_I      = 10
BATCH_F      = 256
GAMMA        = 1e-4             # L1 weight regularisation
BETA         = 0.9              # RMSprop momentum
EPSILON      = 1e-4             # RMSprop stability
VAL_FRAC     = 0.15
PLOT_RESULTS = True

# Model selection
VISIBLE_MODEL = "bernoulli"     # "bernoulli" | "nb"

# Bernoulli model params
BINARIZE_THRESHOLD = "median"     # "median" | "zero"

# NB model params
COUNT_SCALE    = 1000
THETA_INIT_LOG = 0.0

# Paths
DATA_PATH = "data/raw/TimeSeries_countsuL_clean.csv"
if (VISIBLE_MODEL=="nb"):   OUTPUT_DIR = f"results/{VISIBLE_MODEL}_L{N_HIDDEN}"
else:  OUTPUT_DIR = f"results/{VISIBLE_MODEL}_{BINARIZE_THRESHOLD}_L{N_HIDDEN}"