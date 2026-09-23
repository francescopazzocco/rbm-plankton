# Indice materiale_presentazione/

Copia curata da caricare su Claude Design assieme a `SCHELETRO_Presentazione.md`.
Fonte: `results/` del repo, verificata il 2026-09-23 contro quanto generato al momento
(ogni figura qui sotto e' identica byte per byte al file in `results/` indicato).

**Modello finale: NB-Sigmoid (`nb_sigmoid`), L=6, split `shuffled`, seed migliore = seed_9.**
Tutti i plot delle slide 23-29 sono di questo modello e di questo split.

Nota su `nb` vs `nb_sigmoid`: hanno la stessa funzione di energia (hidden Bernoulli con
p(h=1|v) = sigmoide). Differiscono solo nella ricostruzione (media vs campione) e nel monitor,
ma sono **run addestrati separatamente, con pesi diversi**: le figure di uno non valgono per l'altro.

## Documenti di contesto (root)
- `SCHELETRO_Presentazione.md` — scaletta
- `DECISION_LOG.md` — tutti i LOG-XXX citati nella scaletta
- `ARCHITECTURE.md` — slide 12 (layout repo)
- `ROADMAP.md` — slide 32 (future work)
- `results_README.md` — indice dei risultati pubblicati
- `archetype_rbm_comparison.md` — report testuale per slide 28 (**vedi avvertenza sotto**)
- `rbm_model_equations.md` — utile per slide 6-7 (teoria RBM/energy-based)
- `split_strategy.md` — utile per slide 4 e per motivare la scelta shuffled
- `nll_count_scale_issue.md` — utile per sezione D (stabilita' numerica)
- `Data_analysis.md` — utile per slide 3

## Mappatura slide -> file

| Slide | File | Sorgente in `results/` / nota |
|---|---|---|
| 3 Dataset | `03_dataset_fig1_rowsum_timeseries.png`, `03_dataset_fig3_annual_seasonal.png` | `01_exploratory/`. + `03_dataset_EXTRA_fig2_lombscargle.png` (periodogramma, opzionale) |
| 4 Preprocessing | `04_preprocessing_fig5_nan_structure.png`, `04_preprocessing_fig4_distributions.png` | `01_exploratory/` |
| 16 Hidden->Sigmoid | `16_sigmoid_nll_curves_nb_vs_nb_sigmoid_L6.png`, `16_sigmoid_sweep_final_metric_nb_sigmoid.png` | `diagnostics/training_curves/family_comparison_fixed_L/shuffled/nb_vs_nb_sigmoid_L6.png` (curve train/val NB-Bernoulli vs NB-Sigmoid, 10 seed), `04_model_selection/shuffled/individual/`. **Vedi incoerenza slide 16 sotto.** `16_sigmoid_zinb_sigmoid_train_nll_curves.png` e' ZINB-Sigmoid, non il modello finale: solo di contorno |
| 17 Hidden->Softmax (scartato) | `17_softmax_mean_activation_nb_softmax.png`, `17_softmax_state_timeline_nb_softmax.png`, `17_softmax_nb_softmax_train_nll_curves.png`, `17_softmax_zinb_softmax_train_nll_curves.png` | `02_model_analysis/hidden/{mean_activation,state_timeline}/shuffled/nb_softmax.png`, ora L4-L10: da L7 in su solo ~6 unita' vengono usate, le altre restano morte |
| 19 Validazione multi-seed | `19_multiseed_sweep_final_metric_overview.png` | `04_model_selection/shuffled/` (bande = min/max su 10 seed) |
| 21 Monitor training / NaN | `21_monitor_sweep_training_curves.png`, `21_monitor_nan_eval_bars.png`, `21_monitor_nan_eval_timeseries.png`, `21_monitor_split_comparison.png` | `21_monitor_sweep_training_curves.png` rigenerata come griglia 2x5 (era una riga illeggibile; `plot_sweep_curves` in `visualization.py`), identica a `results/diagnostics/training_curves/all_families_by_L/shuffled/val_metric_by_family.png` (ripubblicata). I tre di valutazione NaN/split hanno ancora le etichette vecchie ("NB", "Bernoulli-median"): non rigenerati di proposito, perche' la valutazione Gibbs non e' seedata e i numeri pubblicati cambierebbero |
| 23 Pattern analysis | `23_pattern_coverage_nb_sigmoid_L6.png`, `23_pattern_histogram_nb_sigmoid_L6.png`, `23_pattern_timeline_nb_sigmoid_L6.png` | `hidden/nb_pattern_frequency/shuffled/sigmoid_L6.png`, `hidden/patterns/shuffled/pattern_*_nb_sigmoid_L6_threshold.png` (34 pattern distinti su 64) |
| 24 Attivazione media / stato dominante | `24_activation_mean_activation_nb_sigmoid.png`, `24_activation_state_timeline_nb_sigmoid.png`, `24_activation_dominant_state_L6.csv` | pannelli per ogni L: usare quello L=6. Il csv ha una colonna per famiglia, quella giusta e' `nb_sigmoid` |
| 25 Stackplot temporale | `25_stackplot_nb_sigmoid_L6.png` | `hidden/hidden_stackplot/shuffled/nb_sigmoid_L6.png`. I buchi bianchi sono periodi senza campionamento (> 7 giorni) |
| 26 Overlap archetipi | `26_overlap_heatmap_nb_sigmoid_L6.png` | `archetype/overlap_heatmap/shuffled/` |
| 27 Distanza archetipi | `27_distance_heatmap_nb_sigmoid_L6.png` | `archetype/distance_heatmap/shuffled/` (similarita' coseno) |
| 28 Confronto RBM vs archetipi | `28_closest_archetype_nb_sigmoid_L6.png` + `archetype_rbm_comparison.md` | `archetype/archetype_closest_rbm/shuffled/` |
| 29 Qualita' ricostruzione | `29_reconstruction_nb_sigmoid_L6_seed9.png`, `29_reconstruction_compare_generation_L6_seed0.png` | `reconstruction_plots/` (gitignored, solo locale). Il primo: NB-Sigmoid L6 shuffled seed_9, test set shuffled. Il secondo: tutte le famiglie shuffled L6 seed_0 (seed fisso per tutte, non il migliore di ciascuna) |
| 30 Confronto famiglie | `30_family_comparison_sweep_final_metric_overview.png` + `_bernoulli_median`/`_bernoulli_zero`/`_zinb` | `04_model_selection/shuffled/` |

## Incoerenze tra scaletta e dati (da decidere nel testo delle slide)

- **Slide 16 — "Sigmoid batte NB-Bernoulli"**: i dati attuali non lo mostrano. Val NLL, 10 seed,
  shuffled: L6 NB-Bernoulli 0.4443 ± 0.0150 vs NB-Sigmoid 0.4435 ± 0.0134; L7 0.4373 vs 0.4377;
  L8 0.4320 vs 0.4349. Differenza ~0.001, un decimo della dispersione tra seed; le curve di
  training si sovrappongono (`16_sigmoid_nll_curves_nb_vs_nb_sigmoid_L6.png`). Coerente con
  l'energia identica. La scelta di Sigmoid va motivata altrimenti (ricostruzione in media-campo,
  stabilita' con PCD), non come vittoria in NLL.
- **Slide 18 — "L in {3..7}, esclusione di L=10 per divergenza" (LOG-012)**: lo sweep shuffled
  attuale ha run convergenti fino a L=10 per NB-Bernoulli, BB e NB-Softmax. La divergenza di
  LOG-012 riguardava un singolo run del setup di allora (split chrono, prima di PCD, LOG-016).
- **Slide 28 — `archetype_rbm_comparison.md`**: report scritto a mano (2026-05-22) sui run
  `nb`/`zinb` **chrono**; lo script che lo alimenta ha quelle famiglie hardcoded. Non descrive
  il modello finale: usarlo solo per le conclusioni qualitative, oppure va riscritto su
  `nb_sigmoid` shuffled.

## Altri gap — plot citati nello scheletro mai esistiti (indipendenti dallo split)

- **Slide 8-9**: traiettorie theta L=7 vs L=10. Dati grezzi in
  `artifacts/models/*/{chrono,shuffled}/L*/seed_*/rbm_training_curves.csv` (colonna `theta_mean`),
  nessun plot dedicato.
- **Slide 10**: MSE pre/post PCD. Nessuna traccia nel repo; non rigenerabile senza i checkpoint pre-PCD.
- **Slide 14**: confronto fit BB vs NBB. Nessun plot dedicato; il confronto e' in
  `29_family_comparison_*` ma BB usa PLL e NB usa NLL, non direttamente confrontabili.

## Non incluso (valutare a parte)

- `doc/Eyring_2025_Plankton.pdf` — paper coperto da copyright citato per l'anomalia gen-feb 2023
  (slide 31). Non copiato qui (non-negotiable #2 del progetto); caricarlo a mano se serve.
