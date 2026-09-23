# RBM Model Equations

## 1. BB-RBM — Bernoulli–Bernoulli

### 1.1 Binarisation: per-taxon median threshold

Let $v_i^{(t)} \in \mathbb{R}_{\geq 0}$ be the raw count of taxon $i$ on day $t$.
Define the per-taxon threshold as the empirical median over the training set $\mathcal{T}_{\text{train}}$ only:

$m_i = \mathrm{median}\left(\{v_i^{(t)}\}_{t \in \mathcal{T}_{\text{train}}}\right)$

The binarised input fed to the model is:

$\tilde{v}_i^{(t)} = \mathbf{1}\left[v_i^{(t)} > m_i\right] \in \{0, 1\}$

By construction, approximately 50% of training observations satisfy $\tilde{v}_i = 1$ for every taxon.
The biological meaning shifts from presence/absence to **above/below-median abundance**:
$\tilde{v}_i = 1$ means taxon $i$ is present at above-median abundance on day $t$.

A zero threshold would instead yield $\tilde{v}_i = 0$ for nearly all observations of a rare taxon
(e.g. one absent 90% of the time), collapsing the gradient signal for that unit.
The median threshold avoids this degeneracy by ensuring balanced activation across all taxa.

> **Note on data leakage.** $m_i$ is fit on $\mathcal{T}_{\text{train}}$ and then applied to both validation and test sets.
> Val/test observations are binarised against a training-period abundance level, not their own median.
> This has a consequence: if the abundance distribution shifts between train and val (val = 2024, a single contiguous period), $m_i$ may no longer sit near the 
> val-period median, and $\tilde{v}_i$ in the val set carries less information than in the train set.
> The 50/50 balance guarantee holds only on the training set.

---

### 1.2 Energy function and joint distribution

The BB-RBM defines a joint distribution over visible units $\mathbf{v} \in \{0,1\}^D$
and hidden units $\mathbf{h} \in \{0,1\}^L$ via an energy function:

$E(\mathbf{v}, \mathbf{h}) = -\mathbf{a}^\top \mathbf{v} - \mathbf{b}^\top \mathbf{h} - \mathbf{v}^\top W \mathbf{h}$

where $W \in \mathbb{R}^{D \times L}$ is the weight matrix, $\mathbf{a} \in \mathbb{R}^D$ the visible bias,
and $\mathbf{b} \in \mathbb{R}^L$ the hidden bias. The joint probability is:

$p(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} \exp\bigl(-E(\mathbf{v}, \mathbf{h})\bigr)$

$Z = \sum_{\mathbf{v} \in \{0,1\}^D} \sum_{\mathbf{h} \in \{0,1\}^L} \quad \exp\bigl(-E(\mathbf{v}, \mathbf{h})\bigr)$

$Z$ is the partition function. It is a sum over $2^{D+L}$ configurations and is **intractable** for
$D = 83$, $L = 6$  this is the fundamental computational difficulty of RBMs and the reason the true log-likelihood cannot be directly optimised.

---

### 1.3 Conditional distributions

The bipartite graph structure (no intra-layer connections) makes all units in one layer
conditionally independent given the other. The conditionals factor as:

$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i=1}^{D} W_{ij}\, v_i\right)$

$P(v_i = 1 \mid \mathbf{h}) = \sigma\left(a_i + \sum_{j=1}^{L} W_{ij}\, h_j\right)$

where $\sigma(x) = (1 + e^{-x})^{-1}$ is the sigmoid function.
These are the two Gibbs sampling steps used in Contrastive Divergence.

---

### 1.4 Free energy

Marginalising over hidden units yields the **free energy** $\mathcal{F}(\mathbf{v})$, which gives the (unnormalised) log-probability of a visible configuration:

$p(\mathbf{v}) = \frac{e^{-\mathcal{F}(\mathbf{v})}}{Z}, \qquad \mathcal{F}(\mathbf{v}) = -\log \sum_{\mathbf{h} \in \{0,1\}^L} e^{-E(\mathbf{v}, \mathbf{h})}$

Substituting the energy and factoring the sum (each $h_j$ is independent given $\mathbf{v}$):

$\mathcal{F}(\mathbf{v}) = -\mathbf{a}^\top \mathbf{v} - \sum_{j=1}^{L} \mathrm{softplus}\left(b_j + \sum_{i=1}^{D} W_{ij}\, v_i\right)$

where $\mathrm{softplus}(x) = \log(1 + e^x)$. This is **exact** and computable in $O(DL)$ because the $2^L$ terms in the sum reduce to $L$ independent binary marginals.

---

### 1.5 Pseudo-log-likelihood (PLL)

Since $Z$ is intractable, training maximises the **pseudo-log-likelihood** instead,
a tractable proxy introduced by Besag (1975):

$\mathrm{PLL}(\mathbf{v}) = \frac{1}{D} \sum_{i=1}^{D} \log p(v_i \mid \mathbf{v}_{-i})$

**Derivation of $p(v_i \mid \mathbf{v}_{-i})$.** Write the conditional as a ratio of marginals:

$p(v_i \mid \mathbf{v}_{-i}) = \frac{p(\mathbf{v})}{\displaystyle\sum_{k \in \{0,1\}} p(\mathbf{v}^{(i \to k)})}$

where $\mathbf{v}^{(i \to k)}$ denotes $\mathbf{v}$ with unit $i$ clamped to $k$.
Since $Z$ appears in both numerator and denominator, it cancels:

$p(v_i = 1 \mid \mathbf{v}_{-i}) = \frac{e^{-\mathcal{F}(\mathbf{v}^{(i \to 1)})}}{e^{-\mathcal{F}(\mathbf{v}^{(i \to 1)})} + e^{-\mathcal{F}(\mathbf{v}^{(i \to 0)})}} = \sigma\left(-\Delta\mathcal{F}_i\right)$

where the free-energy difference is:

$\Delta\mathcal{F}_i(\mathbf{v}) = \mathcal{F}(\mathbf{v}^{(i \to 1)}) - \mathcal{F}(\mathbf{v}^{(i \to 0)})$

Expanding using the free-energy formula with $s_{ij} = b_j + \sum_{k \neq i} W_{kj}\, v_k$:

$\Delta\mathcal{F}_i = -a_i - \sum_{j=1}^{L} \Bigl[\mathrm{softplus}(s_{ij} + W_{ij}) - \mathrm{softplus}(s_{ij})\Bigr]$

The training objective is the **negative PLL** (NPLL), expressed as binary cross-entropy:

$\mathrm{NPLL} = -\frac{1}{N \cdot D} \sum_{n=1}^{N} \sum_{i=1}^{D} \mathrm{BCE}\left(\sigma\left(-\Delta\mathcal{F}_i^{(n)}\right),\; v_i^{(n)}\right)$

where $\mathrm{BCE}(p, y) = -y \log p - (1-y) \log(1-p)$.

NPLL is **exact** (no sampling) and costs $O(N \cdot D \cdot L)$ per evaluation.
Lower NPLL is better.

---

## 2. NB-RBM — Negative Binomial–Bernoulli

### 2.1 Negative binomial distribution

The negative binomial distribution parametrised by mean $\mu > 0$ and dispersion $\theta > 0$ has PMF:

$p(v\,;\,\mu, \theta) = \frac{\Gamma(v + \theta)}{\Gamma(\theta)\,\Gamma(v + 1)} \left(\frac{\theta}{\theta + \mu}\right)^{\theta} \left(\frac{\mu}{\theta + \mu}\right)^{v}, \qquad v \in \mathbb{Z}_{\geq 0}$

Key properties:

$\mathbb{E}[v] = \mu, \qquad \mathrm{Var}(v) = \mu + \frac{\mu^2}{\theta}$

The variance exceeds the mean for any finite $\theta$ — this is the **overdispersion** relative
to Poisson, which corresponds to the limit $\theta \to \infty$. Small $\theta$ means high overdispersion (heavy-tailed counts); large $\theta$ approaches Poisson behaviour.

Structural zeros are accommodated without modification:

$p(v = 0\,;\,\mu, \theta) = \left(\frac{\theta}{\theta + \mu}\right)^{\theta} > 0$

This is why NB is appropriate for the plankton data, where 25.7% of all values are zero.

---

### 2.2 Log-gamma function

The gamma function and its logarithm are defined for $z > 0$:

$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t}\, dt, \qquad \log\Gamma(z) = \log\int_0^\infty t^{z-1} e^{-t}\, dt$

For positive integers, $\Gamma(n) = (n-1)!$, but $\Gamma$ is well-defined for all $z > 0$ via analytic continuation. This is required here because the raw data (organisms/$\mu$L, values in $[0,\, 0.44]$) is multiplied by COUNT_SCALE $= 1000$, yielding values in $[0,\, 440]$ that are not exact integers. Using $\log\Gamma$ instead of $\log(v!)$ allows the NB log-likelihood to be evaluated at non-integer $v$ without rounding, via PyTorch's `torch.lgamma`.

---

### 2.3 NB log-likelihood

Taking the logarithm of the PMF:

$\log p(v_i\,;\,\mu_i, \theta_i) = \underbrace{\log\Gamma(v_i + \theta_i) - \log\Gamma(\theta_i) - \log\Gamma(v_i + 1)}_{\text{combinatorial term}}$
$\quad + \underbrace{\theta_i \log\frac{\theta_i}{\theta_i + \mu_i} + v_i \log\frac{\mu_i}{\theta_i + \mu_i}}_{\text{likelihood term}}$

The combinatorial term accounts for the multiplicity of outcomes; the likelihood term encodes the probability mass at $v_i$ given $\mu_i$ and $\theta_i$.

---

### 2.4 Conditional distributions

The hidden conditional is **identical** to BB-RBM. The visible conditional is replaced by NB:

$P(h_j = 1 \mid \mathbf{v}) = \sigma\left(b_j + \sum_{i=1}^{D} W_{ij}\, v_i\right)$

$p(v_i \mid \mathbf{h}) = \mathrm{NB}\left(\mu_i(\mathbf{h}),\; \theta_i\right), \qquad \mu_i(\mathbf{h}) = \exp\left(a_i + \sum_{j=1}^{L} W_{ij}\, h_j\right)$

The exp link ensures $\mu_i > 0$ for all $\mathbf{h}$. Here $a_i$ is a **log-mean baseline** (not a logit parameter as in BB-RBM): $\mu_i = e^{a_i}$ when $\mathbf{h} = \mathbf{0}$.

Each taxon has its own dispersion parameter $\theta_i = \exp(\log\theta_i) > 0$, learned independently from $W$, $\mathbf{a}$, $\mathbf{b}$ via autograd on the positive-phase NLL.

---

### 2.5 CD gradient — weighted residual

The CD gradient for $W$ and $\mathbf{a}$ is derived from the score of the NB log-likelihood with respect to the linear predictor $\eta_i = a_i + \sum_j W_{ij} h_j$:

$r_i(\mathbf{v}, \mathbf{h}) = \frac{\partial \log p(v_i \mid \mathbf{h})}{\partial \eta_i} = \frac{\theta_i\,(v_i - \mu_i)}{\mu_i + \theta_i}$

This is the **weighted residual** — it reduces to the ordinary residual $(v_i - \mu_i)$ in the Poisson limit $\theta_i \to \infty$, and downweights large residuals as $\theta_i \to 0$.

The CD-1 parameter updates for a mini-batch of size $B$ are:

$\nabla_{W_{ij}}\,\mathcal{L} \approx \frac{1}{B} \sum_{n=1}^{B} \Bigl[h_j^{+(n)}\, r_i^{+(n)} - h_j^{-(n)}\, r_i^{-(n)}\Bigr]$

$\nabla_{a_i}\,\mathcal{L} \approx \frac{1}{B} \sum_{n=1}^{B} \Bigl[r_i^{+(n)} - r_i^{-(n)}\Bigr]$

$\nabla_{b_j}\,\mathcal{L} \approx \frac{1}{B} \sum_{n=1}^{B} \Bigl[P(h_j{=}1 \mid \mathbf{v}^{+(n)}) - P(h_j{=}1 \mid \mathbf{v}^{-(n)})\Bigr]$

Superscript $+$ denotes the positive phase (real data), $-$ the negative phase (fantasy particles).
The $\mathbf{b}$ update uses expected activations rather than samples to reduce gradient variance.

---

### 2.6 Persistent Contrastive Divergence — PCD-1

**Why CD-1 fails at $L \geq 5$.** CD-1 restarts the Gibbs chain from a real data point every mini-batch. The chain runs for only one step before the negative-phase sample is taken, so it never moves far from the data distribution. 

When the model has multiple well-separated modes (bloom vs non-bloom community states), the negative-phase samples are biased toward high-data-density regions and the gradient estimate for $\nabla \log Z$ is systematically wrong.

**PCD-1 construction.** Maintain a buffer of $K = 500$ persistent fantasy particles:

$\mathcal{P} = \{\tilde{\mathbf{v}}^{(1)}, \ldots, \tilde{\mathbf{v}}^{(K)}\}, \qquad \text{initialised from } \mathcal{T}_{\text{train}}$

At each mini-batch of size $B \leq K$, apply one Gibbs step to a random subset:

$\text{1.} \quad \text{Sample } \mathcal{I} \subset [K],\; |\mathcal{I}| = B \quad \text{(without replacement)}$

$\text{2.} \quad \tilde{\mathbf{h}}^{(k)} \sim P(\mathbf{h} \mid \tilde{\mathbf{v}}^{(k)};\, \theta_t), \qquad \tilde{\mathbf{v}}^{(k)}_{\text{new}} \sim P(\mathbf{v} \mid \tilde{\mathbf{h}}^{(k)};\, \theta_t), \qquad k \in \mathcal{I}$

$\text{3.} \quad \tilde{\mathbf{v}}^{(k)} \leftarrow \tilde{\mathbf{v}}^{(k)}_{\text{new}} \quad \text{(write back to buffer)}$

$\text{4.} \quad \mathbf{v}^{-(n)} = \tilde{\mathbf{v}}^{(k_n)}_{\text{new}}, \quad \mathbf{h}^{-(n)} = \tilde{\mathbf{h}}^{(k_n)} \quad \text{(supply negative phase)}$

Because particles are never reset, they migrate between modes across parameter updates.
Over training, the buffer approximates samples from $P_{\theta_t}(\mathbf{v})$ at the current parameters $\theta_t$, providing a better estimate of $\nabla_\theta \log Z$.

The condition $K \geq B$ ensures no particle is reused within the same mini-batch draw.
The written-back particles approximate $P_{\theta_t}(\mathbf{v})$; the slight inconsistency introduced by the parameter update $\theta_t \to \theta_{t+1}$ between steps 2 and 3 is the standard PCD approximation and is generally considered acceptable.
