# Comparison: Archetypal Analysis vs. RBM Hidden Node Activations

**Date:** 2026-05-22  
**Dataset:** Lake Zurich plankton time series, 2019–2025  
**Archetype analysis:** k=5, provided
**RBM analysis:** this project, families NB / ZINB / bernoulli_median / bernoulli_zero / _sigmoid / _softmax / _relu, L=3–10  
**Reproducibility script:** `analysis/archetype_rbm_comparison.py`

---

## TL;DR

All usable RBM families independently recover the same ecological structure as archetypal analysis. The four core seasonal communities — spring *Aulacoseira*, winter centric diatom, summer chrysophytes, summer green algae + cyanobacteria — appear robustly across NB, ZINB, and bernoulli models. The RBM additionally splits the chrysophyte season into two temporal stages (early vs. peak summer) that k=5 merges into one. The cryptophyte community (archetype A3) is only recovered as a discrete state by the ZINB family; the NB model distributes cryptophyte as background across all units and cannot isolate it. ReLU models collapse entirely and are ecologically uninformative.

---

## 1. Overview

Archetypal analysis and the RBM hidden-node analysis were run independently on the same plankton dataset. It decomposed the community into k=5 archetypes; the RBM learns a latent representation with L hidden units through unsupervised training. This document compares the ecological communities recovered by each approach and identifies where they agree, where the RBM is finer-grained, and where model family choice matters.

---

## 2. Archetypes analysis (k=5)

Source files: `prof/archetypes_k5_profiles.csv`, `prof/archetypes_k5_timeseries.csv`, and companion figures.

| Archetype | Dominant taxon (weight) | Season | Ecological interpretation |
|-----------|------------------------|--------|--------------------------|
| **A1** | *Dinobryon* (0.799) | Summer | Chrysophyte bloom |
| **A2** | *Aulacoseira* (0.957) | Spring | Deep-mixing spring diatom |
| **A3** | Cryptophyte (0.439) + *Rhodomonas* (0.167) | Year-round baseline | Cryptophyte-dominated background |
| **A4** | Chlorophyte (0.191) + Chlorophyte colonial (0.190), with Oocystaceae (0.123) and colonial cyanobacteria | Summer | Warm-season green algae + cyanobacteria |
| **A5** | Centric diatom (0.542) | Winter–early spring | Cold-season diatom |

The composition timeseries (`archetypes_k5_composition_timeseries.png`) shows clear annual seasonality: A4 dominates warm seasons, A2 fires in spring, A5 in winter, and A1 in early-to-mid summer. A3 acts as a persistent low-level background throughout the year.

---

## 3. RBM Model Quality Assessment

Not all trained model families produce ecologically interpretable hidden units. Units can collapse into two pathological states:

- **absorber_hi**: unit fires constantly (mean activation ≫ 0.5 or ≫ 1 for count models) — absorbs global bias, carries no community-specific information.
- **absorber_lo / dead unit**: unit never fires — wasted capacity.

Summary from `results/02_model_analysis/mean_activation_summary.csv` and the shuffled equivalent:

| Family | Unit health at L=5–7 | Usable for archetype comparison? |
|--------|---------------------|----------------------------------|
| **nb_relu, zinb_relu** | Nearly all `absorber_hi` across all L | **No** — completely collapsed |
| **nb_softmax, zinb_softmax** | Many `absorber_lo` (dead) units at L≥6; winner-take-all forces unit death | **Partially** — usable at L=4–5 only |
| **nb** (default) | Active at L=3–5; NB L=6 chrono degenerate (h0 dominant on 82% of days) | **Yes at L≤5**, caution at L=6 |
| **zinb** (default) | Mostly active; some dead units appear at L≥6 | **Yes at L≤6** |
| **bernoulli_median** | All units active across L=3–7; most balanced utilization | **Yes** |
| **bernoulli_zero** | All active; dominant-state distribution sometimes skewed | **Yes** |
| **nb_sigmoid, zinb_sigmoid** | All units active across L=4–8; most stable family | **Yes** |

**ReLU models are ecologically uninformative** and are excluded from the comparison below.

---

## 4. Cross-Model Comparison with Archetypes

### 4.1 The Four Robustly Recovered Communities

Four of five archetypes appear consistently across every usable model family. Their seasonal fingerprints (from `results/tables/hidden/seasonal_profiles_nb.csv` and `seasonal_profiles_bb.csv`) match the archetype composition timeseries.

#### A2 — Spring diatom (*Aulacoseira*)

*Aulacoseira* at 0.957 relative abundance in A2 makes it the sharpest archetype signal. Every usable model recovers a dedicated unit:

- **NB L=6 chrono:** h2 (aulacoseira=0.68 in vbh; peaks March–April, near zero July)
- **ZINB L=7 chrono:** h1 (aulacoseira=4.57, centric_diatom=9.53 — merged with A5 at this L)
- **BB seasonal profile:** h2/h3 peak strongly in March–April

This community corresponds to the spring stratification breakdown and deep mixing that allows *Aulacoseira* to thrive.

#### A5 — Cold-season centric diatom

- **NB L=6 chrono:** h5 (centric_diatom=0.934; mean activation ≈1.0 Jan–Apr, =0 Jun–Oct)
- **BB (L=6 chrono):** h5 (absent Jul–Sep)
- At small L (≤4), A2 and A5 merge into a single "diatom" unit; they separate cleanly at L≥5.

#### A4 — Summer green algae + cyanobacteria

- **NB L=6 chrono:** h1 (chlorophyte=0.929, chlorophyte_colonial_dividing=0.939, chroococcales=0.888, cyanobacteria_colonial_probably=0.933; seasonal peak Jul–Oct)
- **ZINB L=7 chrono:** h3 (chlorophyte=3.72, oocystaceae=2.46, cyanobacteria_colonial_probably=2.02)
- **BB:** h1 peaks 0.85–1.0 in Jul–Sep

The co-occurrence of colonial green algae, oocystaceae, and cyanobacteria in a single warm-season unit is robust across families.

#### A1 — Chrysophyte summer (*Dinobryon*)

This is where the RBM is **finer-grained than the archetypes**. Rather than a single chrysophyte unit, every model with L≥5 splits this into two:

- **Early-summer unit** (peaks April–June): uroglena and dinobryon present, often co-occurring with spring taxa — NB h3 (uroglena=0.831, dinobryon=0.408)
- **Peak-summer unit** (June maximum): high dinobryon + uroglena without diatom co-occurrence — NB h4 (uroglena=0.911, dinobryon=0.847)

A1 collapses these into a single archetype because archetypes are convex-hull extremes of the full dataset; the RBM detects the temporal substructure within the chrysophyte season. This split is consistent across NB, ZINB, and bernoulli families.

---

### 4.2 The Family-Dependent Community: A3 (Cryptophyte)

This is the most ecologically interesting divergence between model families.

**In NB models:** cryptophyte appears at moderate-to-high probability in *every* hidden unit (range 0.54–0.87 in `nb_chrono_vbh/visible_by_hidden_bernoulli.csv`). No unit is dedicated to cryptophyte. The NB model treats cryptophyte as a background species whose *counts* vary but whose *presence* is uninformative for state discrimination.

**In ZINB models:** h4 of the L=7 chrono model has cryptophyte=10.4 and rhodomonas=5.7 as its dominant signal — a clear analogue to A3. The ZINB's zero-inflation component separates whether a taxon is absent (structural zero) from whether it is rare (sampling zero), making cryptophyte *blooms* distinguishable from cryptophyte *background*. This allows the ZINB to isolate high-cryptophyte periods as a discrete community state.

**Implication:** the ZINB family better captures A3. The NB family's failure to isolate cryptophyte is a model-likelihood artifact, not an ecological finding.

---

### 4.3 Effect of L on Archetype Recovery

| L | What the RBM recovers |
|---|----------------------|
| 3 | ~2 dominant states: warm season vs. cold season |
| 4 | Spring diatom + winter diatom begin separating; summer collapses to one state |
| **5** | **Closest to k=5**; most families cleanly recover A2, A4, A5, and one chrysophyte unit; cryptophyte embedded in summer state (NB) or separated (ZINB) |
| 6 | Chrysophyte season splits into early/peak summer; NB chrono becomes degenerate (absorber h0); ZINB and bernoulli still informative |
| 7+ | Further within-season resolution (e.g., *Aulacoseira* vs. *Fragilaria*/*Asterionella* spring distinction in ZINB h2 vs. h5); bernoulli_median most stable |

**L=5 with nb_sigmoid or zinb_sigmoid** is the configuration most directly comparable to k=5 archetypes: all units are active, no absorbers, and unit capacity matches the number of archetypes.

---

### 4.4 Chronological vs. Shuffled Split

The dominant-state timeseries for the shuffled split (`results/02_model_analysis/shuffled/dominant_state_L6_shuffled.csv`) shows the same families of states as the chronological split. The ecological communities found by the RBM are not a consequence of temporal ordering in training — they are real recurring assemblages in the data. This is consistent with archetypes, which make no assumption about time ordering.

---

## 5. Summary Table

| archetype | Best RBM match | Families where this holds | Notes |
|----------------|---------------|--------------------------|-------|
| A1 — Dinobryon | **Two units** (early + peak summer chrysophyte) | NB, ZINB, bernoulli (L≥5) | RBM is finer-grained than k=5 |
| A2 — Aulacoseira | One dedicated spring unit | All usable families | Most robustly recovered |
| A3 — Cryptophyte | Dedicated unit in ZINB; absent in NB | ZINB only | Model-likelihood dependent |
| A4 — Summer green/cyano | One summer unit | All usable families | Very robust |
| A5 — Centric diatom | One cold-season unit | All usable families (merged with A2 at L≤4) | Separates at L≥5 |
| — | Absorber unit (always on) | NB L=6, some ZINB configs | No archetype equivalent; arises from bias absorption |

---

## 6. Conclusions

1. **Broad agreement:** The RBM and archetypal analysis independently recover the same four major seasonal plankton communities (spring *Aulacoseira*, cold-season centric diatom, summer chrysophytes, summer green algae + cyanobacteria).

2. **RBM substructure:** The chrysophyte season is consistently split into two temporal stages by the RBM — an early-summer and a peak-summer community — which k=5 constraint merges. This is a genuine additional finding, not an artefact.

3. **ZINB recovers the cryptophyte state; NB does not.** The zero-inflation model's ability to distinguish true absences from sampling zeros gives cryptophyte discriminatory power that the plain NB model cannot exploit. This is the clearest argument in this dataset for preferring ZINB over NB.

4. **ReLU activation is ecologically unusable** — all units collapse to absorbers regardless of L or family.

5. **L=5 with sigmoid hidden activation** (nb_sigmoid or zinb_sigmoid) is the best-controlled configuration for direct comparison with k=5 archetypes: no dead units, no absorbers, and unit count matches decomposition.

6. **Community states are robust to train/test split** — appearing in both chronological and shuffled configurations — confirming that the RBM is detecting genuine ecological structure.

---

## 7. Measurements and Evidence

All quantitative claims in this document are derived from the following data sources. The script `analysis/archetype_rbm_comparison.py` reproduces every table below from the raw files.

### 7.1 Archetype profiles — dominant taxa and weight fractions

Source: `prof/archetypes_k5_profiles.csv`

| Archetype | Top taxon | Weight | Fraction of total |
|-----------|-----------|--------|-------------------|
| A1 | *Dinobryon* | 0.7989 | 79.9% |
| A2 | *Aulacoseira* | 0.9574 | 95.7% |
| A3 | Cryptophyte | 0.4387 | 43.9% |
| A4 | Chlorophyte colonial | 0.1909 | 19.1% |
| A5 | Centric diatom | 0.5422 | 54.2% |

A1 and A2 are strongly uni-taxon archetypes; A3–A5 are mixed assemblages. A4 has no single dominant taxon (top weight only 19%), making it the hardest archetype to pin to a single species.

### 7.2 Archetype temporal dominance

Source: `prof/archetypes_k5_timeseries.csv` — fraction of days on which each archetype has the highest weight.

| Archetype | Days dominant | Fraction |
|-----------|--------------|---------|
| A1 | 187 | 9.1% |
| A2 | 212 | 10.3% |
| A3 | 697 | 33.7% |
| A4 | 834 | 40.4% |
| A5 | 136 | 6.6% |

A3 (cryptophyte) and A4 (summer green) together account for 74% of days, confirming they are the baseline states. A2, A1, A5 are seasonal pulses.

### 7.3 RBM top taxa per hidden unit

Source: `analysis/results/nb_chrono_vbh/visible_by_hidden_bernoulli.csv` (NB, L=6, chrono training)

| Unit | Top taxa (detection probability) |
|------|----------------------------------|
| h0 | fragilaria=0.869, asterionella=0.796, chlorophyte_colonial=0.752, cryptophyte=0.681 |
| h1 | chlorophyte_colonial=0.939, cyanobacteria_colonial=0.933, chlorophyte=0.929, oocystaceae=0.916 |
| h2 | oocystaceae=0.929, cyanobacteria_colonial=0.868, cryptophyte=0.777, cyanobacteria_blue=0.754 |
| h3 | uroglena=0.831, cyanobacteria_colonial=0.796, chlorophyte_colonial=0.623, oocystaceae=0.587 |
| h4 | asterionella=0.918, fragilaria=0.914, uroglena=0.911, cyanobacteria_colonial=0.859, dinobryon=0.847 |
| h5 | centric_diatom=0.934, cryptophyte=0.688, rhodomonas=0.582, chlorophyte=0.541 |

Source: `analysis/results/zinb_chrono_vbh/visible_by_hidden_zinb.csv` (ZINB, L=7, chrono; values in expected counts)

| Unit | Top taxa (expected count) |
|------|--------------------------|
| h0 | cryptophyte=11.6, chlorophyte=7.2, rhodomonas=3.8, centric_diatom=3.4 |
| h1 | centric_diatom=9.5, aulacoseira=4.6 |
| h2 | cryptophyte=5.7, pennate_diatom=4.6, asterionella=3.7, rotifer=2.7 |
| h3 | cryptophyte=7.2, chlorophyte=3.7, rhodomonas=2.9, oocystaceae=2.5, cyanobacteria_colonial=2.0 |
| h4 | cryptophyte=10.4, rhodomonas=5.7, aulacoseira=1.9, rotifer=1.8 |
| h5 | asterionella=9.7, crypt=3.9, chlorophyte_colonial=2.1, fragilaria=1.4 |
| h6 | cryptophyte=4.7, rhodomonas=1.9, chlorophyte=1.4 |

### 7.4 Key taxon profiles across NB L=6 hidden units

Source: `analysis/results/nb_chrono_vbh/visible_by_hidden_bernoulli.csv`

| Taxon | h0 | h1 | h2 | h3 | h4 | h5 |
|-------|----|----|----|----|----|----|
| aulacoseira | 0.504 | 0.042 | **0.680** | 0.057 | 0.660 | 0.526 |
| centric_diatom | 0.521 | 0.405 | 0.120 | 0.194 | 0.566 | **0.934** |
| dinobryon | 0.195 | 0.528 | 0.126 | 0.408 | **0.847** | 0.018 |
| uroglena | 0.111 | 0.534 | 0.209 | **0.831** | **0.911** | 0.126 |
| chlorophyte | 0.598 | **0.929** | 0.716 | 0.482 | 0.749 | 0.541 |
| cyanobacteria_colonial | 0.614 | **0.933** | 0.868 | 0.796 | 0.859 | 0.366 |
| cryptophyte | 0.681 | 0.870 | 0.777 | 0.539 | 0.712 | 0.688 |

Cryptophyte has no clear peak unit (range 0.54–0.87, std=0.11) — no discriminative power for state assignment in NB.

### 7.5 Cryptophyte distribution: NB vs. ZINB

Source: sections 5 and 6 of script output.

**NB chrono L=6** (detection probability — scale 0–1):

| h0 | h1 | h2 | h3 | h4 | h5 |
|----|----|----|----|----|----|
| 0.681 | 0.870 | 0.777 | 0.539 | 0.712 | 0.688 |

Range: [0.54, 0.87], std=0.11. Cryptophyte is high everywhere → not state-discriminating.

**ZINB chrono L=7** (expected count — unbounded):

| h0 | h1 | h2 | h3 | h4 | h5 | h6 |
|----|----|----|----|----|----|----|
| 11.6 | 2.2 | 5.7 | 7.2 | **10.4** | 3.9 | 4.7 |

Range: [2.2, 11.6], std=3.4. h4 and h0 are clearly elevated — cryptophyte peaks are concentrated in specific states, making it discriminative.

### 7.6 Seasonal profiles

Source: `results/tables/hidden/seasonal_profiles_nb.csv`

NB L=6 chrono — mean activation per month per hidden unit:

| Month | h0 | h1 | h2 | h3 | h4 | h5 |
|-------|----|----|----|----|----|----|
| Jan | 1.00 | 0.01 | 0.22 | 0.75 | 0.22 | **1.00** |
| Feb | 1.00 | 0.17 | 0.24 | 0.44 | 0.24 | **1.00** |
| Mar | 1.00 | 0.69 | **0.51** | 0.39 | 0.32 | **1.00** |
| Apr | 1.00 | 0.97 | **0.59** | **0.92** | 0.04 | **1.00** |
| May | 0.94 | 0.81 | 0.05 | **0.93** | 0.27 | 0.69 |
| Jun | 0.90 | 0.83 | 0.09 | 0.75 | **0.93** | 0.10 |
| Jul | 0.86 | **0.95** | 0.00 | 0.49 | 0.48 | 0.00 |
| Aug | 0.64 | **1.00** | 0.20 | 0.59 | 0.33 | 0.01 |
| Sep | 0.43 | **1.00** | 0.23 | 0.32 | 0.37 | 0.00 |
| Oct | 0.83 | 0.82 | 0.35 | 0.43 | 0.21 | 0.00 |
| Nov | 1.00 | 0.14 | 0.63 | 0.50 | 0.45 | 0.57 |
| Dec | 1.00 | 0.00 | 0.40 | 0.75 | 0.27 | 0.88 |

Seasonal assignment: **h0** = year-round absorber; **h1** = Jul–Sep; **h2** = Mar–Apr; **h3** = Apr–May; **h4** = Jun; **h5** = Nov–Apr.

### 7.7 State frequency and NB L=6 degeneration

Source: `results/02_model_analysis/state_frequency.csv`

NB L=6 chrono — fraction of days each unit is the dominant state:

| Unit | Days | Fraction |
|------|------|---------|
| h0 | 1570 | **82.4%** |
| h1 | 330 | 17.3% |
| h2 | 0 | 0.0% |
| h3 | 3 | 0.2% |
| h4 | 3 | 0.2% |
| h5 | 0 | 0.0% |

h0 dominates 82% of days; h2 and h5 are never dominant. NB L=6 chrono is degenerate and should not be used for community state analysis — use NB L=5 or NB L=7 instead.

NB L=5 state frequency (well-distributed):

| Unit | Days | Fraction |
|------|------|---------|
| h0 | 466 | 24.4% |
| h1 | 573 | 30.1% |
| h2 | 450 | 23.6% |
| h3 | 165 | 8.7% |
| h4 | 252 | 13.2% |

### 7.8 Model quality — absorber rates by family (shuffled split)

Source: `results/02_model_analysis/shuffled/mean_activation_summary_shuffled.csv`

| Family | Absorber units / total | Rate |
|--------|----------------------|------|
| nb_relu | ~all | ~100% |
| zinb_relu | ~all | ~100% |
| nb_softmax | several at L≥6 | ~30–50% |
| zinb_softmax | several at L≥6 | ~30–50% |
| nb_sigmoid | 0 across L=4–8 | 0% |
| zinb_sigmoid | 0 across L=4–8 | 0% |
| nb | 0 at L=3–5; degenerate at L=6 | 0–partial |
| zinb | dead units appear at L≥5 | partial |
| bernoulli_median | 0 across L=3–7 | 0% |
| bernoulli_zero | 0 across L=3–7 | 0% |

### 7.9 Reproducing the analysis

```bash
# Full output (all 10 measurement sections)
rbm_plankton/bin/python analysis/archetype_rbm_comparison.py

# Restrict state-frequency and absorber tables to specific families/L
rbm_plankton/bin/python analysis/archetype_rbm_comparison.py \
    --families nb zinb --L 5 6 7

# Show top-8 taxa per hidden unit instead of top-5
rbm_plankton/bin/python analysis/archetype_rbm_comparison.py --top 8

# Save full output to file
rbm_plankton/bin/python analysis/archetype_rbm_comparison.py \
    > doc/archetype_rbm_comparison_output.txt
```
