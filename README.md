# NLP Assignment 1 — Phase 1: Cleaning IMDB Movie Reviews for Sentiment Analysis

## 1. Topic and Dataset
Our project classifies movie reviews as positive or negative (binary sentiment analysis). We use the **IMDB Dataset of 50K Movie Reviews**, which has 50,000 English reviews with perfectly balanced labels.

| Item | Value |
| :--- | :--- |
| **Source** | Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL (`[http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)`) |
| **Download used by our script** | Public GitHub mirror of the 50K CSV (`DATA_URL` in `src/config.py`) |
| **Size** | 50,000 reviews (25,000 positive, 25,000 negative); 49,582 after dropping 418 exact duplicates |
| **Licence** | Released by the authors for research use; paper cited |
| **Future target (Phase 2)** | `label`: `1` = positive, `0` = negative |

## 2. Repository Structure
- `src/config.py` — Paths, dataset URL, and constants
- `src/data.py` — Script to download and verify the raw IMDB dataset (without committing raw data)
- `src/nltk_setup.py` — Downloads required NLTK tokenizers, stop words, and WordNet[cite: 51]
- `src/explore.py` & `notebooks/01_exploration.ipynb` — Corpus statistics, Zipf plot, and messy text analysis[cite: 51]
- `src/regex_patterns.py`, `src/tokenization.py` & `notebooks/02_tokenization_regex.ipynb` — Regex, NLTK, and spaCy tokenizer comparison[cite: 51]
- `src/cleaning.py`, `src/run_pipeline.py` & `notebooks/03_cleaning_pipeline.ipynb` — End-to-end normalization and cleaning pipeline[cite: 51]

## 3. Reproducibility (How to Run from Scratch)
Run the following commands in your terminal to set up the environment, download the raw dataset, and run the complete cleaning pipeline[cite: 51, 52]:

    git clone [https://github.com/Khaqan-J/NLP-Assignment-01-pipeline.git](https://github.com/YOUR-USERNAME/NLP-Assignment-01-pipeline.git)
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
| **Member A (Team Lead)** | Setup, data loading, reproducibility (`src/config.py`, `src/data.py`, `src/nltk_setup.py`, `README.md`)[cite: 51] | `feat/setup-data`[cite: 51] |
| **Member B** | Exploration and corpus statistics (`src/explore.py`, `notebooks/01_exploration.ipynb`)[cite: 51] | `feat/exploration`[cite: 51] |
| **Member C** | Tokenization and regular expressions (`src/regex_patterns.py`, `src/tokenization.py`, `notebooks/02_tokenization_regex.ipynb`)[cite: 51] | `feat/tokenization-regex`[cite: 51] |
| **Member D** | Cleaning pipeline (`src/cleaning.py`, `src/run_pipeline.py`, `notebooks/03_cleaning_pipeline.ipynb`)[cite: 51] | `feat/cleaning-pipeline`[cite: 51] |