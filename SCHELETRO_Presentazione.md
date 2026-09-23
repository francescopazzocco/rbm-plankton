# Scheletro presentazione finale — RBM-plankton

## A. Intro & motivazione (slide 1-2)

### 1 Copertina
Titolo, autori/relatori,

### 2 Obiettivo
- contesto del progetto (serie temporali di plankton + RBM)
- Motivazione scientifica: perche' un modello generativo per le comunita' planctoniche; domanda di ricerca

## B. Dati & pipeline (slide 3-5)

### 3 Dataset // 01_exploratory/fig1 & fig3
- Dataset interessante: enorme lavoro di raccolta alle spalle
- Ma con forti limitazioni: missing entry sparsi, molto rumore
- Setup tipico in cui si vuole estrarre il massimo dell'informazione nonostante i problemi -> motivazione per usare metodi di machine learning
- Fonte dati, periodo temporale, taxa monitorati

### 4 Preprocessing // 01_exploratory/fig5 & fig4
- Drop righe all-zero e NaN (LOG-001, LOG-002)
- Split train/val cronologico 85/15 (LOG-004)

### 5 Normalizzazione e binarizzazione // supeflua/missplaced(?): binarizzazione è solo per RBM poi perchè non row-norm lo si può dire in 4
- Perche' niente normalizzazione per riga (LOG-003)
- Soglia di binarizzazione = mediana per-taxon (LOG-005)

## C. Teoria RBM generale (slide 6-7)

### 6 Energy-based model
- Intuizione, visible/hidden units, contrastive divergence

### 7 Perche' una RBM
- Ci aspettiamo che estragga comunque una buona mappa delle feature latenti del dataset, nonostante missing e rumore
- Perche' RBM su dati di conteggio sparsi/overdispersi con zeri strutturali
- (VAE come alternativa: solo in lavoro futuro)

## D. Numerical stability — cosa abbiamo controllato per farlo girare stabile (slide 8-11)

### 8 Clamp su eta (NBB-RBM)
- Bound su mu per restare in float32 range (LOG-010)
- Perche' non float64: throughput GPU consumer ~1/32

### 9 Diagnosi instabilita' NB-RBM a L>=5
- Slow Gibbs mixing su distribuzioni multimodali, non gradient explosion (LOG-015)
- Plot: traiettorie theta L=7 vs L=10

### 10 Fix: Persistent Contrastive Divergence
- PCD per attraversare le energy barrier tra i modi (LOG-016)
- Plot: MSE pre/post PCD

### 11 Clamp ReLU + centralizzazione costanti
- Clamp attivazioni ReLU [0,5] per scale mismatch conteggi/hidden (LOG-020)
- Centralizzazione di tutte le costanti numeriche in `_constants.py` (LOG-023)
- Bug di documentazione emerso nell'audit: clamp dichiarato [-5,5] vs implementato [-10,10]

## E. Struttura attuale del codice (slide 12, jolly — nessun vincolo di posizione, basta che ci sia)

### 12 Layout del repository
- `src/` (libreria) vs `scripts/` (entry point) (LOG-028)
- `results/` come tier pubblicato con `MANIFEST.json` (LOG-029, quello sul results tier)
- `scripts/analysis/` diviso in `hidden/` / `archetype/` / `reconstruction/` (LOG-030)

## F. Come abbiamo ricavato la best architecture (slide 13-17)

### 13 Punto di partenza: BB-RBM
- RBM puramente discreta (visible+hidden Bernoulli): la piu' facile da implementare (LOG-006, LOG-007)

### 14 Visible continuo: Negative Binomial
- NBB-RBM: Negative Binomial visible units per conteggi overdispersi con zeri strutturali (LOG-006)
- Una volta garantita la stabilita' (PCD, vedi D) si puo' espandere al continuo anche gli hidden
- Plot: confronto fit BB vs NBB

### 15 Hidden continui: ReLU instabile
- Primo tentativo: ReLU, numericamente instabile anche col clamp (LOG-020, gia' visto in D)
- -> si provano alternative limitate: Sigmoid e Softmax

### 16 Hidden -> Sigmoid
- Bounded, PCD-safe, nessuna unita' morta (LOG-021)
- NLL sostanzialmente pari a NB-Bernoulli (stessa energia): scelta motivata da stabilita' e ricostruzione mean-field, non da un guadagno in NLL
- Plot: curve NLL NB-Bernoulli vs NB-Sigmoid

### 17 Hidden -> Softmax: non adatta a ricostruire, ma informativa
- NLL piu' alta di Sigmoid/Bernoulli -> poco adatta a ricostruire (LOG-022)
- Softmax = one-hot: forza pattern di attivazione, soprattutto a L alto
- Multi-seed: in media se ne usano 5-7 -> suggerisce 5-7 pattern latenti reali nel dataset
- Plot: attivazioni / state timeline / curve NLL softmax

## G. Perche' credere ai risultati che genera (slide 18-21)

### 18 Selezione del range L valido
- L in {3..7}: esclusione di L=10 per divergenza dinamica non recuperabile (LOG-012) // attenzione: sweep shuffled attuale converge fino a L=10 (vedi INDEX)

### 19 Validazione statistica multi-seed
- N=10 seed per la scelta di L (LOG-013)
- Plot: bande di varianza multi-seed

### 20 Selezione di n_hidden
- n_hidden = 6 per tutte le famiglie di modello (LOG-017)

### 21 Monitor di training e dati mancanti
- PLL per BB-RBM, NLL per NBB-RBM come diagnostica primaria (LOG-011)
- Saturation monitor per bias absorbers / collasso binario (LOG-011)
- Gestione NaN nel test set via imputazione a osservazione parziale (LOG-018)

## H. Risultati finali del modello selezionato (slide 22-30)

### 22 Il modello finale
- NB-Sigmoid, L=6, split shuffled
- Riepilogo delle decisioni che ci hanno portato li'

### 23 Pattern analysis sulle hidden units
- L'informazione non sta nei singoli nodi ma in configurazioni on-off
- Sigmoid L=6: 10 pattern coprono ~80% di tutti i pattern appresi (da verificare sulla figura coverage)
- Coerente con softmax: sigmoid e' meno "brutale" e tiene un'analisi piu' fine
- Alcuni pattern riconducibili alla fenomenologia del plankton, altri a bias del dataset / rumore sistematico di raccolta
- (`hidden_pattern_analysis.py`)

### 24 Attivazione media e stato dominante
- Attivazione media per hidden unit (`hidden_mean_activation.py`)
- Stato dominante per timestep (`hidden_dominant_state.py`)

### 25 Evoluzione temporale degli stati
- Stackplot temporale delle hidden units / community states (`rbm_hidden_stackplot.py`)

### 26 Overlap con gli archetipi
- Coverage stagionale, overlap tra stati del modello e archetipi noti (`overlap_archetypes_rbm.py`)

### 27 Distanza dagli archetipi
- Distanza tra stati ricostruiti e archetipi di riferimento (`distance_archetypes_rbm.py`)

### 28 Confronto diretto RBM vs archetipi
- Scatter dell'archetipo piu' vicino, confronto complessivo (`archetype_closest_rbm_scatter.py`, `archetype_rbm_comparison.py`)

### 29 Qualita' di ricostruzione del modello finale
- Metriche di fit sul modello selezionato (`use_trained_rbm.py`)

### 30 Confronto tra famiglie di modello
- Sweep finale delle metriche, split per famiglia (`compare_model_reconstructions.py`)

## I. Limiti & lavoro futuro (slide 31-32)

### 31 Limiti
- Anomalia gennaio-febbraio 2023, chiusa come probabile evento ecologico reale (LOG-019)
- Non tutti i pattern appresi sono per forza ecologia: parte puo' essere bias/rumore sistematico

### 32 Lavoro futuro
- Provare un VAE per fare le stesse analisi
- Hidden units Gaussiane mai esplorate fino in fondo (LOG-007)
- Voci Next / Future work di ROADMAP.md

## J. Conclusioni (slide 33-34)

### 33 Sintesi
- Contributi principali del progetto

### 34 Take-away
- Messaggio finale
