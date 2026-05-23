"""
_constants.py - Shared numeric constants for all RBM model classes
===================================================================
Every value here appeared previously as an inline literal scattered across
nb_rbm.py, zinb_rbm.py, bernoulli_rbm.py, base_rbm.py, and
_hidden_monitors.py.  Centralising them ensures that a change to one value
is reflected everywhere, and that the reasoning behind each choice is
written down once rather than inferred from context.

Constants are grouped by role, not by file.
"""


# ---------------------------------------------------------------------------
# Numerical stability — NB / ZINB visible units
# ---------------------------------------------------------------------------

# Minimum value enforced on theta = exp(log_theta) via .clamp(min=...).
# Purpose: prevents theta reaching zero, which makes the NB distribution
# degenerate (variance collapses to the mean; the dispersion degree of freedom
# vanishes).  Also keeps lgamma(V + theta) and log(theta / ...) finite.
# Why 1e-4: large enough to keep lgamma gradients non-zero in float32, small
# enough not to constrain a taxon that genuinely has near-Poisson dispersion.
THETA_CLAMP_MIN = 1e-4

# Additive floor inside log() calls and shared denominators.
# Purpose: prevents log(0) = -inf and division-by-zero when mu or theta drift
# close to zero during early training.
# Why 1e-8: safely above float32 denormal range (~1.2e-38) while being
# negligible relative to typical mu values (O(1) to O(1000) after COUNT_SCALE).
# Used in _nb_log_prob, _nb_residual, _zinb_log_prob, _zinb_residual, and for
# clamping the empirical data mean before log-initialising the visible bias a.
LOG_PROB_EPS = 1e-8

# Maximum value for the log-mean predictor eta = a + H @ W.T before exp().
# Purpose: exp(eta) overflows float32 at eta > ~88; even values in the tens
# produce means far outside the data range and destabilise the CD chain.
# Why 10.0: exp(10) ≈ 22026, which is ~50x the largest observed scaled count
# in this dataset (~444 after COUNT_SCALE=1000), providing a firm ceiling
# while leaving room for the model to explore the natural data range.
ETA_CLAMP_MAX = 10.0

# Bounds applied to log_theta and logit_pi after each gradient step.
# Purpose: prevent the autograd update from sending these parameters to ±∞,
# where gradients vanish and the parameter is effectively frozen for the rest
# of training (a form of irreversible saturation).
# Why ±10:
#   log_theta  in [-10, 10]  <=>  theta  in [4.5e-5, 22026]  — covers the full
#     practical dispersion range from near-Poisson (large theta) to near-zero
#     (very overdispersed); beyond this the NB degenerates in either direction.
#   logit_pi   in [-10, 10]  <=>  pi     in [4.5e-5, 0.9999]  — similarly
#     spans the full inflation range without pinning the parameter at 0 or 1.
LOG_PARAM_CLAMP_MIN = -10.0
LOG_PARAM_CLAMP_MAX = 10.0

# Upper clamp for ReLU / Rectified-Gaussian hidden unit activations.
# Purpose: count-scale visible units (V up to ~444 after COUNT_SCALE=1000)
# push the pre-activation b + V @ W into the hundreds, overwhelming the h²/2
# restoring force of the Gaussian-ReLU energy (which assumes normalised inputs).
# The clamp enforces the intended energy ceiling externally.
# Why 5.0: allows meaningful continuous activation range while preventing
# runaway amplification; chosen to match the O(1) scale assumed by the
# Gaussian-ReLU energy function.
RELU_HIDDEN_CLAMP_MAX = 5.0

# Minimum value for the empirical visible-unit probability in BernoulliRBM
# before converting to log-odds for bias initialisation: a = log(p / (1-p)).
# Purpose: a taxon that is always 0 or always 1 in the training data would
# give log(0) or log(∞); clamping to [PROB_CLAMP_MIN, 1-PROB_CLAMP_MIN]
# keeps the initial bias finite.
# Why 1e-4: same floor as THETA_CLAMP_MIN; with N~600 training samples,
# 1e-4 < 1/N so it only activates for genuinely constant taxa.
PROB_CLAMP_MIN = 1e-4


# ---------------------------------------------------------------------------
# Numerical stability — hidden unit monitoring
# ---------------------------------------------------------------------------

# Activation thresholds defining "saturated low" and "saturated high" for
# Bernoulli hidden units.  A unit is saturated low (high) when p(h=1|v) stays
# persistently below SAT_LO (above SAT_HI) across the training set — it is
# effectively always off (on) and carries no information about the input.
# Why 0.1 / 0.9: symmetric 10% margins; units outside [0.1, 0.9] change their
# binary output for fewer than 10% of inputs, making them near-useless.
# These values are also used by visualization.py to shade activation plots —
# changing them here automatically updates the legend thresholds.
BERNOULLI_SAT_LO = 0.1
BERNOULLI_SAT_HI = 0.9

# Log clamp for the softmax entropy computation in SoftmaxHiddenMonitor.
# Purpose: softmax outputs are strictly in (0,1) in exact arithmetic, but
# float32 underflow can produce exact zeros for very unlikely categories;
# clamping before log prevents -inf in the entropy sum.
# Why 1e-10: much smaller than LOG_PROB_EPS because this term is used only
# for monitoring (no gradient flows through it), so a tighter floor is safe.
SOFTMAX_ENTROPY_EPS = 1e-10


# ---------------------------------------------------------------------------
# Training defaults — shared across BernoulliRBM, NB_RBM, ZINB_RBM
# ---------------------------------------------------------------------------

# Per-epoch multiplicative decay on the learning rate: lr *= LR_DECAY each epoch.
# Why 0.998: over 1000 epochs, 0.998^1000 ≈ 0.135, so the effective lr at the
# end of training is ~13.5% of the initial value.  Keeps learning active for
# most of training and anneals only in the final quarter, which helps avoid
# oscillating around sharp minima without cutting off gradient flow too early.
DEFAULT_LR_DECAY = 0.998

# L1 regularisation coefficient on W (and a, b in BernoulliRBM).
# The update is: W -= gamma * lr * sign(W).
# Why 1e-4: small relative to the CD gradient magnitude (O(1/batch) ≈ 0.004
# with batch=256), so it introduces sparsity pressure without dominating the
# CD signal.  Larger values collapse weights to zero prematurely.
DEFAULT_GAMMA = 1e-4

# Initial and final batch sizes for the quadratic batch-size schedule.
# batch_size(epoch) = int(BATCH_I + (BATCH_F - BATCH_I) * ((epoch-1)/(E-1))^2)
# Why BATCH_I=10: small early batches give noisier but larger-variance gradient
# estimates, which help escape bad local minima in the first few hundred epochs.
# Why BATCH_F=256: standard GPU-friendly power-of-two; with N~600 training
# samples, 256 covers ~43% of the data per batch, giving stable gradient
# estimates at convergence.  Larger batches did not improve final val metrics.
DEFAULT_BATCH_I = 10
DEFAULT_BATCH_F = 256

# Number of random mini-batches drawn per epoch (with replacement).
# Total sample-steps per epoch ≈ N_BATCHES * mean(batch_size).
# Why 20: with the batch schedule above, 20 batches gives roughly
# 20 * 128 ≈ 2560 sample-steps per epoch on average — approximately four
# passes through the training set — decoupled from N so that the step count
# does not change if the dataset size changes.
DEFAULT_N_BATCHES = 20

# RMSprop momentum (exponential moving average coefficient for squared gradients).
# s_t = BETA * s_{t-1} + (1 - BETA) * g_t^2
# Why 0.9: Hinton's original RMSprop recommendation; gives an effective
# "window" of 1/(1-beta) = 10 steps, short enough to adapt to changing
# curvature during early training without losing memory of the gradient scale.
DEFAULT_BETA = 0.9

# RMSprop denominator floor: update = lr * g / sqrt(s + EPS).
# Why 1e-4: deliberately larger than the Adam/TF default of 1e-8.  CD
# gradient magnitudes are small (O(1/batch) ~ 0.004), so a larger floor
# prevents the adaptive denominator from amplifying tiny, noisy gradients
# into large steps during early training when s is near zero.
DEFAULT_RMSPROP_EPS = 1e-4

# Fraction of the main learning rate used for the theta (dispersion) and
# pi (zero-inflation) parameter updates via autograd.
# lr_theta = lr_pi = lr * LR_PARAM_MULTIPLIER  (when not explicitly set)
# Why 0.1: theta and pi gradients involve digamma functions whose magnitudes
# can be several times larger than the CD gradient; a 10x smaller lr keeps
# them on the same effective update scale as W and prevents the dispersion
# from overshooting on early epochs when the NB fit is poor.
LR_PARAM_MULTIPLIER = 0.1


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

# Numerator in the Xavier-style weight scale: W ~ N(0, sqrt(FACTOR / (D + L))).
# Why 4.0: standard Glorot normal uses factor=2 (fan_avg), He uses 2/fan_in.
# The factor 4 was chosen empirically for count-scale inputs (V ~ O(100-1000)
# after COUNT_SCALE=1000): larger inputs require smaller initial weights to
# keep the pre-activations b + V @ W in the O(1) range expected by sigmoid /
# Bernoulli hidden units.  With D=83 and L=6, sqrt(4/89) ≈ 0.21; this gives
# pre-activations std ≈ 0.21 * sqrt(D) * mean(V_std) which lands near 1.
XAVIER_SCALE_FACTOR = 4.0
