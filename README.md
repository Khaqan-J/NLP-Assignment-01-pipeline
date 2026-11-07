# NLP Assignment 1 — Phase 1: Cleaning IMDB Movie Reviews for Sentiment Analysis

## 1. Topic and Dataset
Our project classifies movie reviews as positive or negative (binary sentiment analysis). We use the **IMDB Dataset of 50K Movie Reviews**, which has 50,000 English reviews with perfectly balanced labels.

| Item | Value |
| :--- | :--- |
| **Source** | Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL (`http://ai.stanford.edu/~amaas/data/sentiment/`) |
| **Download used by our script** | Public GitHub mirror of the 50K CSV (`DATA_URL` in `src/config.py`) |
| **Size** | 50,000 reviews (25,000 positive, 25,000 negative); 49,582 after dropping 418 exact duplicates |
| **Licence** | Released by the authors for research use; paper cited |
| **Future target (Phase 2)** | `label`: `1` = positive, `0` = negative |

## 2. Repository Structure
- `src/config.py` — Paths, dataset URL, and constants
- `src/data.py` — Script to download and verify the raw IMDB dataset (without committing raw data)
- `src/nltk_setup.py` — Downloads required NLTK tokenizers, stop words, and WordNet
- `src/explore.py` & `notebooks/01_exploration.ipynb` — Corpus statistics, Zipf plot, and messy text analysis
- `src/regex_patterns.py`, `src/tokenization.py` & `notebooks/02_tokenization_regex.ipynb` — Regex, NLTK, and spaCy tokenizer comparison
- `src/cleaning.py`, `src/run_pipeline.py` & `notebooks/03_cleaning_pipeline.ipynb` — End-to-end normalization and cleaning pipeline

## 3. Reproducibility (How to Run from Scratch)
Run the following commands in your terminal to set up the environment, download the raw dataset, and run the complete cleaning pipeline:

    git clone https://github.com/Khaqan-J/NLP-Assignment-01-pipeline.git
    cd NLP-Assignment-01-pipeline
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python -m src.nltk_setup
    python -m src.data
    python -m src.run_pipeline

## 4. Team Members & Responsibilities
| Member | Main Responsibility | Branch |
| :--- | :--- | :--- |
| **Member A (Khaqan) (Team Lead)** | Setup, data loading, reproducibility (`src/config.py`, `src/data.py`, `src/nltk_setup.py`, `README.md`) | `feat/khaqan-setup-data` |
| **Member B (Alisha)** | Exploration and corpus statistics (`src/explore.py`, `notebooks/01_exploration.ipynb`) | `feat/alisha-exploration` |
| **Member C (Maryam)** | Tokenization and regular expressions (`src/regex_patterns.py`, `src/tokenization.py`, `notebooks/02_tokenization_regex.ipynb`) | `feat/maryam-tokenization-regex` |
| **Member D (Mati)** | Cleaning pipeline (`src/cleaning.py`, `src/run_pipeline.py`, `notebooks/03_cleaning_pipeline.ipynb`) | `feat/mati-cleaning-pipeline` |

## 5. Exploration Findings (Task 2 - Alisha)
Code: `src/explore.py`. Notebook: `notebooks/01_exploration.ipynb` (run it to reproduce). Figures are saved in `reports/figures/`.

| Statistic | Value |
| :--- | :--- |
| Documents | 50,000 (no missing values, 418 exact duplicates) |
| Words per review | mean 231.2, median 173, min 4, max 2,470 |
| Target distribution | 25,000 positive, 25,000 negative |
| Tokens / types (5,000-review sample) | 1,412,078 / 53,488 (type-token ratio 0.038) |
| Hapax legomena | 25,147 (47.0% of types) |

**Zipf's law:** the log-log rank/frequency plot is close to a straight line, and the 100 most frequent types cover 55.3% of all tokens.

**Messy text found (share of reviews):** HTML `<br />` 58.4%, ALL-CAPS words 22.7%, explicit ratings such as 8/10 9.3%, non-ASCII characters 9.3%, extra spaces 5.0%, elongated words 3.1%, URLs 0.4%.
