# Scheletro presentazione finale — RBM-plankton

## A. Intro & motivazione (slide 1-2)

### 1 Copertina
Titolo, autori/relatori,

### 2 Obiettivo
- contesto del progetto (serie temporali di plankton + RBM)
- Motivazione scientifica: perche' un modello generativo per le comunita' planctoniche; domanda di ricerca

## B. Dati & pipeline (slide 3-5)

### 3 Dataset // 01_exploratory/fig1 & fig3
- Fonte dati, periodo temporale, taxa monitorati

### 4 Preprocessing // 01_exploratory/fig5 & fig4
- Drop righe all-zero e NaN (LOG-001, LOG-002)
- Split train/val cronologico 85/15 (LOG-004)

### 5 Normalizzazione e binarizzazione // supeflua/missplaced(?): binarizzazione è solo per RBM poi perchè non row-norm lo si può dire in 4
- Perche' niente normalizzazione per riga (LOG-003)
- Soglia di binarizzazione = mediana per-taxon (LOG-005)

## C. Teoria RBM generale + perche' RBM e non VAE (slide 6-8)

### 6 Energy-based model
- Intuizione, visible/hidden units, contrastive divergence

### 7 RBM
- Perche' RBM su dati di conteggio sparsi/overdispersi con zeri strutturali
- Energy function esplicita vs prior gaussiano latente + reparametrization trick

### 8 (opzionale) Un VAE su questi dati //togli superflua la mettiamo in future work eg. provare con VAE
- Cosa avrebbe richiesto di diverso

## D. Numerical stability — cosa abbiamo controllato per farlo girare stabile (slide 9-12)

### 9 Clamp su eta (NBB-RBM)
- Bound su mu per restare in float32 range (LOG-010)
- Perche' non float64: throughput GPU consumer ~1/32

### 10 Diagnosi instabilita' NB-RBM a L>=5
- Slow Gibbs mixing su distribuzioni multimodali, non gradient explosion (LOG-015)
- Plot: traiettorie theta L=7 vs L=10

### 11 Fix: Persistent Contrastive Divergence
- PCD per attraversare le energy barrier tra i modi (LOG-016)
- Plot: MSE pre/post PCD

### 12 Clamp ReLU + centralizzazione costanti
- Clamp attivazioni ReLU [0,5] per scale mismatch conteggi/hidden (LOG-020)
- Centralizzazione di tutte le costanti numeriche in `_constants.py` (LOG-023)
- Bug di documentazione emerso nell'audit: clamp dichiarato [-5,5] vs implementato [-10,10]

## E. Struttura attuale del codice (slide 13, jolly — nessun vincolo di posizione, basta che ci sia)

### 13 Layout del repository
- `src/` (libreria) vs `scripts/` (entry point) (LOG-028)
- `results/` come tier pubblicato con `MANIFEST.json` (LOG-029)
- `scripts/analysis/` diviso in `hidden/` / `archetype/` / `reconstruction/` (LOG-030)

## F. Come abbiamo ricavato la best architecture (slide 14-18)

### 14 Punto di partenza: BB-RBM
- Visible+hidden Bernoulli, tutto discreto (LOG-006, LOG-007)

### 15 Da Bernoulli a Negative Binomial
- NBB-RBM: Negative Binomial visible units per conteggi overdispersi con zeri strutturali (LOG-006)
- Plot: confronto fit BB vs NBB

### 16 Hidden Bernoulli -> ReLU
- Motivazione: community states continui
- Richiedeva il clamp gia' visto in D per non esplodere (LOG-020) — nessun nuovo plot, e' lo stesso fix

### 17 Hidden -> Sigmoid
- PCD-safe, stabile, batte NB-Bernoulli (LOG-021)
- Plot: curva NLL a confronto (questa e' l'evidenza che decide l'architettura finale)

### 18 Hidden -> Softmax (scartato)
- Collassa ad assegnazioni quasi-deterministiche (LOG-022)
- Plot: attivazioni collassate
- Chiusura: la traiettoria discreto->continuo e' emersa empiricamente, non pianificata a tavolino

## G. Perche' credere ai risultati che genera (slide 19-22)

### 19 Selezione del range L valido
- L in {3..7}: esclusione di L=10 per divergenza dinamica non recuperabile (LOG-012)

### 20 Validazione statistica multi-seed
- N=10 seed per la scelta di L (LOG-013)
- Plot: bande di varianza multi-seed

### 21 Selezione di n_hidden
- n_hidden = 6 per tutte le famiglie di modello (LOG-017)

### 22 Monitor di training e dati mancanti
- PLL per BB-RBM, NLL per NBB-RBM come diagnostica primaria (LOG-011)
- Saturation monitor per bias absorbers / collasso binario (LOG-011)
- Gestione NaN nel test set via imputazione a osservazione parziale (LOG-018)

## H. Risultati finali del modello selezionato (slide 23-31)

### 23 Il modello finale
- Famiglia, L, n_hidden scelti
- Riepilogo delle decisioni che ci hanno portato li'

### 24 Pattern analysis sulle hidden units
- Pattern dominanti individuati nelle hidden units (`hidden_pattern_analysis.py`)

### 25 Attivazione media e stato dominante
- Attivazione media per hidden unit (`hidden_mean_activation.py`)
- Stato dominante per timestep (`hidden_dominant_state.py`)

### 26 Evoluzione temporale degli stati
- Stackplot temporale delle hidden units / community states (`rbm_hidden_stackplot.py`)

### 27 Overlap con gli archetipi
- Coverage stagionale, overlap tra stati del modello e archetipi noti (`overlap_archetypes_rbm.py`)

### 28 Distanza dagli archetipi
- Distanza tra stati ricostruiti e archetipi di riferimento (`distance_archetypes_rbm.py`)

### 29 Confronto diretto RBM vs archetipi
- Scatter dell'archetipo piu' vicino, confronto complessivo (`archetype_closest_rbm_scatter.py`, `archetype_rbm_comparison.py`)

### 30 Qualita' di ricostruzione del modello finale
- Metriche di fit sul modello selezionato (`use_trained_rbm.py`)

### 31 Confronto tra famiglie di modello
- Sweep finale delle metriche, split per famiglia (`compare_model_reconstructions.py`)

## I. Limiti & lavoro futuro (slide 32-33)

### 32 Limiti
- Anomalia gennaio-febbraio 2023, chiusa come probabile evento ecologico reale (LOG-019)

### 33 Lavoro futuro
- Direzioni lasciate aperte: hidden units Gaussiane mai esplorate fino in fondo (LOG-007)
- Voci Next / Future work di ROADMAP.md

## J. Conclusioni (slide 34-35)

### 34 Sintesi
- Contributi principali del progetto

### 35 Take-away
- Messaggio finale
