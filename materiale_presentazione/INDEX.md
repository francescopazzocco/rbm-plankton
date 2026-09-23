# Indice materiale_presentazione/

Copia curata da caricare su Claude Design assieme a `SCHELETRO_Presentazione.md`.
Fonte: repo dopo il reorg (`results/` con split `chrono/`/`shuffled/`).

**Protocollo finale confermato: split `shuffled`** (ordine temporale abbandonato di proposito —
piu' preciso nella minimizzazione della NLL, pratica standard e difendibile). Famiglia finale:
`nb`/`nb_sigmoid` (stessa architettura, nome legacy — confermato) con L=6.
Tutti i plot delle slide 17-31 sotto usano quindi lo split `shuffled`, non `chrono`.

## Documenti di contesto (root)
- `SCHELETRO_Presentazione.md` — scaletta
- `DECISION_LOG.md` — tutti i LOG-XXX citati nella scaletta
- `ARCHITECTURE.md` — slide 13 (layout repo)
- `ROADMAP.md` — slide 33 (future work)
- `results_README.md` — indice dei risultati pubblicati
- `archetype_rbm_comparison.md` — report testuale per slide 29 (confronto complessivo RBM vs archetipi)
- `rbm_model_equations.md` — utile per slide 6-8 (teoria RBM/energy-based)
- `split_strategy.md` — utile per slide 4 (split cronologico) e per motivare la scelta shuffled in sezione D/G
- `nll_count_scale_issue.md` — utile per sezione D (stabilita' numerica)
- `Data_analysis.md` — utile per slide 3

## Mappatura slide -> file

| Slide | File | Nota |
|---|---|---|
| 3 Dataset | `03_dataset_fig1_rowsum_timeseries.png`, `03_dataset_fig3_annual_seasonal.png` | + `03_dataset_EXTRA_fig2_lombscargle.png` (non citato nello scheletro, periodogramma — valuta se serve) |
| 4 Preprocessing | `04_preprocessing_fig5_nan_structure.png`, `04_preprocessing_fig4_distributions.png` | |
| 17 Hidden->Sigmoid | `17_sigmoid_zinb_sigmoid_train_nll_curves.png`, `17_sigmoid_sweep_final_metric_nb_sigmoid.png` | il secondo e' l'unico plot esplicitamente chiamato `nb_sigmoid` in tutto il repo (shuffled/individual) |
| 18 Hidden->Softmax (scartato) | `18_softmax_nb_softmax_train_nll_curves.png`, `18_softmax_zinb_softmax_train_nll_curves.png`, `18_softmax_mean_activation_nb_softmax.png`, `18_softmax_state_timeline_nb_softmax.png` | gia' shuffled (softmax non esiste su chrono) |
| 20 Validazione multi-seed | `20_multiseed_sweep_final_metric_overview.png` (shuffled) | bande di varianza N=10 seed gia' incluse — non serve generare un plot nuovo |
| 22 Monitor training / NaN | `22_monitor_nan_eval_bars.png`, `22_monitor_nan_eval_timeseries.png`, `22_monitor_split_comparison.png`, `22_monitor_sweep_training_curves.png` (shuffled) | i tre nan_eval non hanno split chrono/shuffled distinti in results/ |
| 24 Pattern analysis | — | **gap, vedi sotto** |
| 25 Attivazione media / stato dominante | `25_activation_mean_activation_nb.png`, `25_activation_state_timeline_nb.png`, `25_activation_dominant_state_L6.csv` (shuffled) | il csv non ha un plot dedicato — solo tabella |
| 26 Stackplot temporale | `26_stackplot_nb_L6_CHRONO_FALLBACK.png` | **gap, vedi sotto** — copiato comunque come riferimento visivo, ma e' chrono non shuffled |
| 27 Overlap archetipi | `27_overlap_heatmap_nb_L6.png` (shuffled) | |
| 28 Distanza archetipi | `28_distance_heatmap_nb_L6.png` (shuffled) | |
| 29 Confronto RBM vs archetipi | `29_closest_archetype_nb_L6.png` (shuffled) + `archetype_rbm_comparison.md` | |
| 30 Qualita' ricostruzione | `30_reconstruction_nb_L6_seed0.png`, `30_reconstruction_compare_generation_L6_seed0.png` | **da verificare**: `reconstruction_plots/` non ha sottocartelle chrono/shuffled — non e' chiaro da quale split vengano questi due file (il `shuffle-0` nel nome sembra indicare l'indice di ripetizione, non lo split dati) |
| 31 Confronto famiglie | `31_family_comparison_sweep_final_metric_overview.png` (shuffled, split per famiglia, e' il plot decisivo) + `_bernoulli_median`/`_bernoulli_zero`/`_zinb` (shuffled) per il dettaglio | |

## Gap reali — plot che servirebbero ma non esistono per lo split shuffled

- **Slide 24 (pattern analysis)**: `results/02_model_analysis/hidden/patterns/shuffled/` contiene
  solo `pattern_*_nb_softmax_L7_winner.*` (l'ablation softmax), non l'equivalente generico
  per `nb_sigmoid_L6`. La versione L6 completa esiste solo su chrono
  (`patterns/chrono/pattern_histogram_nb_L6_threshold.png`). Va rigenerata con
  `hidden_pattern_analysis.py --family nb --split shuffled` (o come si chiama l'argomento split)
  prima di poter usare la slide 24 con dati coerenti col resto del deck.
- **Slide 26 (stackplot)**: `hidden_stackplot/` ha solo `chrono/nb_L6.png`, nessuna versione
  shuffled. Va rigenerata con `rbm_hidden_stackplot.py` sullo split shuffled. Ho copiato la
  versione chrono come `26_stackplot_nb_L6_CHRONO_FALLBACK.png` solo come riferimento visivo/di
  layout — **non usarla come dato reale nella slide finale**.

## Altri gap — plot citati nello scheletro mai esistiti (indipendenti dallo split)

- **Slide 9-10**: traiettorie theta L=7 vs L=10. Dati grezzi ci sono
  (`artifacts/models/nb/chrono/L7/seed_*/rbm_training_curves.csv`,
  `artifacts/models/*/shuffled/L10/...`) ma nessun plot renderizzato — va creato da questi CSV.
- **Slide 11**: MSE pre/post PCD. Nessuna traccia nel repo (probabilmente un plot diagnostico
  ad hoc fatto durante il debug, mai salvato in `results/`). Non rigenerabile senza i checkpoint
  pre-PCD, se esistono ancora.
- **Slide 15**: confronto fit BB vs NBB. Nessun plot dedicato trovato — puoi costruirlo da
  `results/diagnostics/sweep/shuffled/sweep_nb_diagnostics.png` + l'equivalente per bernoulli,
  oppure toglierlo dallo scheletro.

## Non incluso (valutare a parte)

- `doc/Eyring_2025_Plankton.pdf` — il paper coperto da copyright citato per l'anomalia gen-feb 2023
  (slide 32). Non l'ho copiato qui per lo stesso motivo per cui non va su git (non-negotiable #2
  del progetto); se ti serve per generare slide 32 caricalo separatamente e a mano.
