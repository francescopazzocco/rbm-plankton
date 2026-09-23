# Split Strategy: Chronological vs Shuffled

## What question does each answer?

| Strategy | Answers |
|---|---|
| **Chronological** | *Does the model generalise to an unseen future year?* Tests if community states learned from past years (2019–2023) predict patterns in a held-out year (2024). |
| **Shuffled** | *What is the full set of community states in the dataset?* Tests the model's ability to discover latent structure from the entire 5-year record, without temporal hold-out. |

## Empirical comparison (NB-RBM L=6, N=10 seeds)

| Metric | Chronological | Shuffled |
|---|---|---|
| Val NLL (mean ± std) | 0.5574 ± 0.0072 | 0.4443 ± 0.0150 |
| Avg hidden units >0.5 | 3.25 | 2.55 |
| Unique 6-bit patterns | 34 | 26 |

## L selection outcome

| Strategy | Optimal L | Reasoning |
|---|---|---|
| Chronological | 6 | L=6→7 adds 0.2σ (noise). Temporal generalisation ceiling at L=6. |
| Shuffled | 7 | L=6→7 adds ~1.6% val NLL improvement and ~18 new hidden patterns. |

## Pros and cons

### Chronological split

| Pros | Cons |
|---|---|
| Honest estimate of temporal generalisation | Val NLL conflates model quality with distribution shift |
| Ecologically meaningful: answers "do past patterns predict future?" | 2024-specific community states cannot appear in training |
| L selection is conservative, less prone to overfitting | Misses ~15–20 latent patterns that L=7 captures (shuffled) |
| Standard in time-series ML evaluation | Val set is a single contiguous window — less robust estimate |

### Shuffled split

| Pros | Cons |
|---|---|
| Maximises pattern discovery: model sees full temporal range | Val NLL is optimistic (leaks temporal structure into training) |
| L selection reflects data's intrinsic dimensionality, not temporal ceiling | Not a valid estimate of future-year generalisation |
| Lower val NLL variance across seeds (easier task) | Standard in i.i.d. ML, not time-series — requires justification |
| L=7 shows real additional structure (~44 patterns vs ~26 at L=6) | — |

## Recommendation

- **Use shuffled** if the goal is **community state discovery** (what patterns exist in the lake across 5 years?). Select L=7.
- **Use chronological** if the goal is **temporal generalisation** (can patterns learned from past predict the future?). Select L=6.
- For a paper: present chronological as primary (defensible, honest) and shuffled as a robustness / pattern-recovery supplement.
