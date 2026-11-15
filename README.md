# End-to-End NLP Pipeline: IMDB Movie Reviews
**Course:** Introduction to Natural Language Processing  
**Repository Tags:** `assignment-1` (Phase 1) | `assignment-2` (Phase 2)

An end-to-end Natural Language Processing pipeline built on the **Stanford Large Movie Review Dataset (50,000 IMDB movie reviews)** for binary sentiment analysis.
* **Phase 1 (`assignment-1`):** Data acquisition, corpus exploration, regular expression cataloguing, tokenizer benchmarking, and an 8-step sentiment-aware text cleaning and normalization pipeline.
* **Phase 2 (`assignment-2`):** Count-based representations (Bag of Words, TF-IDF, N-grams, hand-worked numerical verification), learned neural word embeddings (`Word2Vec` CBOW, `Word2Vec` Skip-gram, `fastText`), intrinsic semantic probing, and unified model packaging for downstream classification.

---

## 1. Team Members & Branch Architecture

| Member | Role | Phase 1 Branch & Tasks (`assignment-1`) | Phase 2 Branch & Tasks (`assignment-2`) |
| :--- | :--- | :--- | :--- |
| **Khaqan Javed** | Team Lead | `feat/khaqan-setup-data`<br>Repo setup, config, automated dataset loader (`src/data.py`), README & release tagging | `feat/khaqan-phase2-packaging`<br>Part D: Model persistence (`src/save_models.py`), `vectorize()` interface (`src/vectorize.py`), notebook `07`, README & `assignment-2` tag |
| **Alisha** | Member | `feat/alisha-exploration`<br>Task 2: Corpus statistics, Zipf's law & length plots, messy text analysis (`src/explore.py`, notebook `01`) | `feat/alisha-embeddings`<br>Part B: `Word2Vec` CBOW (`sg=0`), Skip-gram (`sg=1`), `fastText` training & OOV demonstration (`src/embeddings.py`, notebook `05`) |
| **Maryam** | Member | `feat/maryam-tokenization-regex`<br>Task 3: 7 regex patterns & Regex vs. NLTK vs. spaCy benchmarks (`src/regex_patterns.py`, `src/tokenization.py`, notebook `02`) | `feat/maryam-probing-numerical`<br>Part A4 & Part C: Toy numerical TF-IDF/bigram verification & semantic probing (`src/probing.py`, notebook `06`) |
| **Mati** | Member | `feat/mati-cleaning-pipeline`<br>Task 4 & 5: 8-step normalization, sentiment-safe stopwords & lemmatization (`src/cleaning.py`, `src/run_pipeline.py`, notebook `03`) | `feat/mati-count-representations`<br>Part A1–A3: Bag of Words, TF-IDF weight analysis & N-gram growth trade-off (`src/count_reps.py`, notebook `04`) |

---

## 2. Repository Structure

```text
NLP-Assignment-01-pipeline/
├── data/
│   ├── raw/                              # Raw IMDB_Dataset.csv (50,000 reviews, gitignored)
│   └── processed/                        # Cleaned imdb_cleaned.csv (49,582 deduplicated reviews, gitignored)
├── models/                               # Saved Phase 2 models (generated via python -m src.save_models)
│   ├── tfidf.pkl                         # Fitted TfidfVectorizer (50,000 max features)
│   ├── cbow.model                        # Trained Word2Vec CBOW model (100-d)
│   ├── skipgram.model                    # Trained Word2Vec Skip-gram model (100-d)
│   └── fasttext.model                    # Trained fastText sub-word model (100-d)
├── notebooks/
│   ├── 01_exploration.ipynb              # Phase 1 (Alisha): Corpus statistics, Zipf plot & noise analysis
│   ├── 02_tokenization_regex.ipynb       # Phase 1 (Maryam): Regex catalog & tokenizer benchmarks
│   ├── 03_cleaning_pipeline.ipynb        # Phase 1 (Mati): 8-step cleaning examples & vocab reduction
│   ├── 04_count_representations.ipynb    # Phase 2 (Mati): BoW, TF-IDF high/low weights & N-gram growth
│   ├── 05_learned_embeddings.ipynb       # Phase 2 (Alisha): CBOW, Skip-gram, fastText & OOV queries
│   ├── 06_probing_and_numerical.ipynb    # Phase 2 (Maryam): Hand-worked numerical check & embedding probes
│   └── 07_packaging_vectorize.ipynb      # Phase 2 (Khaqan): Saved models & vectorize(doc, method) demo
├── reports/
│   ├── cleaning_stats.json               # Before/after vocabulary reduction metrics
│   └── figures/                          # Generated Zipf's law and length/label distribution plots
├── src/
│   ├── config.py                         # Centralized paths and constants
│   ├── data.py                           # Automated IMDB dataset downloader and deduplicator
│   ├── nltk_setup.py                     # NLTK resource initializer
│   ├── explore.py                        # Phase 1 corpus exploration & plotting
│   ├── regex_patterns.py                 # Compiled regular expressions for noise detection & tokenization
│   ├── tokenization.py                   # Regex, NLTK, and spaCy tokenizer comparisons
│   ├── cleaning.py                       # 8-step normalization, negation handling & spaCy lemmatization
│   ├── run_pipeline.py                   # End-to-end Phase 1 cleaning runner
│   ├── count_reps.py                     # Phase 2 Part A1–A3 (BoW, TF-IDF, N-grams)
│   ├── embeddings.py                     # Phase 2 Part B1–B4 (Word2Vec CBOW, Skip-gram, fastText)
│   ├── probing.py                        # Phase 2 Part A4 & C1–C3 (Toy verification, similarity, analogy)
│   ├── save_models.py                    # Phase 2 Part D1 (Trains and saves all representations once)
│   └── vectorize.py                      # Phase 2 Part D2 (Unified vectorize(raw_doc, method) interface)
├── requirements.txt                      # Python package dependencies
└── README.md                             # Project documentation
```

---

## 3. Environment Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Khaqan-J/NLP-Assignment-01-pipeline.git
cd NLP-Assignment-01-pipeline

# 2. Create and activate a Python 3.12 virtual environment
py -3.12 -m venv .venv
source .venv/Scripts/activate

# 3. Install dependencies and language resources
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m src.nltk_setup
```

---

## 4. Phase 1: Data Acquisition, Exploration, Tokenization & Cleaning (`assignment-1`)

### 4.1 Data Acquisition & Deduplication (Task 1 — Khaqan)
* **Code:** `src/config.py`, `src/data.py`
* Automatically downloads the 50,000 IMDB movie reviews (`25,000` positive, `25,000` negative) into `data/raw/IMDB_Dataset.csv` and removes `418` exact duplicate reviews to yield **`49,582` unique documents**.

### 4.2 Exploration Findings (Task 2 — Alisha)
* **Code:** `src/explore.py` | **Notebook:** `notebooks/01_exploration.ipynb` | **Figures:** `reports/figures/`

| Statistic | Value |
| :--- | :--- |
| **Documents** | `50,000` (no missing values, `418` exact duplicates) |
| **Words per review** | mean `231.2`, median `173`, min `4`, max `2,470` |
| **Target distribution** | `25,000` positive, `25,000` negative |
| **Tokens / types (5,000-review sample)** | `1,412,078` / `53,488` (type-token ratio `0.038`) |
| **Hapax legomena** | `25,147` (`47.0%` of types) |

* **Zipf's law:** The log-log rank/frequency plot is close to a straight line, and the 100 most frequent types cover **`55.3%`** of all tokens.
* **Messy text found (share of reviews):** HTML `<br />` `58.4%`, ALL-CAPS words `22.7%`, explicit ratings such as `8/10` `9.3%`, non-ASCII characters `9.3%`, extra spaces `5.0%`, elongated words `3.1%`, URLs `0.4%`.

### 4.3 Tokenization & Regular Expressions (Task 3 — Maryam)
* **Code:** `src/regex_patterns.py`, `src/tokenization.py` | **Notebook:** `notebooks/02_tokenization_regex.ipynb`
* Benchmarks custom `TOKEN_REGEX`, NLTK `word_tokenize`, and spaCy `en_core_web_sm` on a 2,000-review sample and catalogues 7 regex patterns (`HTML_TAG`, `URL`, `ELONGATION`, `RATING`, `CONTRACTION`, `ALLCAPS`, `EMOTICON`) across all 50,000 reviews.

### 4.4 Cleaning & Normalization Pipeline (Task 4 & 5 — Mati)
* **Code:** `src/cleaning.py`, `src/run_pipeline.py` | **Notebook:** `notebooks/03_cleaning_pipeline.ipynb` | **Stats:** `reports/cleaning_stats.json`
* **8-Step Pipeline:** (1) Unicode NFC & quote/dash normalization, (2) HTML entity unescaping & `<br />` tag stripping, (3) URL removal, (4) Lowercasing (after logging ALL-CAPS count), (5) Elongated character collapsing, (6) Contraction/negation expansion (`n't` -> `not`), (7) Lemmatization & non-alphabetic filtering, and (8) Sentiment-aware stop-word removal (preserving 30 negations and intensifiers).
* Reduces raw vocabulary from `167,400` types (`13,903,009` tokens) down to `83,159` lemmatized types (`-50.32%` reduction) across `49,582` deduplicated reviews.

---

## 5. Phase 2: Word Representations (`assignment-2`)

Building on the cleaned IMDB dataset (`data/processed/imdb_cleaned.csv`: `49,582` documents, `6,219,655` tokens, `98,695` unique word types, average length `125.4` tokens), Phase 2 implements and compares count-based and learned neural word representations.

### 5.1 Part A: Count-Based Representations (Mati & Maryam)
* **Code:** `src/count_reps.py`, `src/probing.py` | **Notebooks:** `notebooks/04_count_representations.ipynb`, `notebooks/06_probing_and_numerical.ipynb`
* **A1. Bag of Words (`CountVectorizer`):** Shape **`(49,582, 98,695)`**, `4,987,316` non-zero entries, **`99.8981%` sparsity**.
* **A2. TF-IDF High vs. Near-Zero Weights (Document 0):**
  * **High weight:** `oz` (`0.4541`, df=`176`), `violence` (`0.1997`, df=`1,693`), `prison` (`0.1836`, df=`629`).
  * **Near zero (common):** `well` (`0.0251`, df=`14,887`), `no` (`0.0242`, df=`16,115`), `one` (`0.0178`, df=`28,208`).
* **A3. N-Gram Vocabulary Growth (`min_df=5`):**

| N-gram Range | Number of Features | Matrix Shape | Sparsity (%) |
| :--- | :--- | :--- | :--- |
| **Unigrams `(1,1)`** | `36,424` | `(49,582, 36,424)` | `99.7293%` |
| **Uni + Bigrams `(1,2)`** | `182,468` | `(49,582, 182,468)` | `99.9186%` |
| **Uni + Bi + Trigrams `(1,3)`** | `210,405` | `(49,582, 210,405)` | `99.9265%` |

* **A4. Hand-Worked Numerical Verification:** Confirmed exact match on the 4-document toy corpus for `"great"` in `D1` (`IDF = 1.223144`, raw `TF-IDF = 2.446287`, L2 `TF-IDF = 0.749377`) and bigram `"great movie"` counts `[1, 1, 0, 0]`.

### 5.2 Part B: Learned Embeddings — Word2Vec & fastText (Alisha)
* **Code:** `src/embeddings.py` | **Notebook:** `notebooks/05_learned_embeddings.ipynb`
* **Hyperparameters:** `vector_size=100`, `window=5`, `min_count=5`, `epochs=5`, `seed=42` (Vocabulary size: **`38,984` words**).
* **Training Runtimes:** `Word2Vec` CBOW (`sg=0`): `13.0s` | `Word2Vec` Skip-gram (`sg=1`): `44.1s` | `fastText`: `65.6s`.
* **OOV Demonstration (`"cinematographyyy"`):** Word2Vec raises `KeyError`, while `fastText` retrieves `cinematography (0.994)`, `cinematograph (0.993)`, `cinematographic (0.944)`, `cinematographer (0.935)`.

### 5.3 Part C: Probing & Comparison (Maryam)
* **Code:** `src/probing.py` | **Notebook:** `notebooks/06_probing_and_numerical.ipynb`

| Word Pair | Expected | CBOW | Skip-gram | fastText |
| :--- | :--- | :--- | :--- | :--- |
| **`film` / `movie`** | Similar | `0.8531` | `0.9008` | `0.8175` |
| **`excellent` / `superb`** | Similar | `0.8802` | `0.8936` | `0.8563` |
| **`horror` / `popcorn`** | Unrelated | `0.3349` | `0.5073` | `0.2869` |
| **`script` / `spaceship`** | Unrelated | `-0.1241` | `0.2672` | `-0.0581` |

* **Word Analogy (`actor - man + woman`):** All three models rank **`actress`** #1 (`CBOW: 0.821`, `Skip-gram: 0.885`, `fastText: 0.855`).

### 5.4 Part D: Packaging & Vectorization Interface (Khaqan)
* **Code:** `src/save_models.py`, `src/vectorize.py` | **Notebook:** `notebooks/07_packaging_vectorize.ipynb`
* `src/save_models.py` saves `models/tfidf.pkl`, `models/cbow.model`, `models/skipgram.model`, and `models/fasttext.model`.
* `vectorize(raw_doc, method)` cleans any raw document and returns its `(50000,)` TF-IDF vector or `(100,)` averaged embedding vector without retraining.

---

## 6. Reproducibility

```bash
# Checkout the submission tag
git checkout assignment-2

# Phase 1: Download raw data and generate cleaned corpus
python -m src.data
python -m src.explore
python -m src.tokenization
python -m src.run_pipeline

# Phase 2: Run count representations, train & save models, run probing, and test vectorize()
python -m src.count_reps
python -m src.save_models
python -m src.probing
python -m src.vectorize
```
