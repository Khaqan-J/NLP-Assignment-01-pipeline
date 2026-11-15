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

## 6. Phase 2: Word Representations (Assignment 2)
Building on the cleaned IMDB dataset (`data/processed/imdb_cleaned.csv`), Phase 2 implements count-based and dense word representations, numerical & semantic probing, and model packaging.

### How to Run Phase 2 Modules
- **Part A — Count Representations (BoW, TF-IDF, N-grams):**
  ```bash
  python -m src.count_reps
  ```
- 3. Part B: Learned Embeddings
Unlike count-based vectors, which are long and sparse, embeddings are short dense vectors learned from the corpus so that words used in similar contexts get similar vectors. We trained all models with gensim on the same tokenized corpus: the cleaned IMDB reviews from Phase 1 (data/processed/imdb_cleaned.csv, column text_clean), split into tokens (49,582 reviews, 6,219,655 tokens, 98,695 unique tokens).
B1 and B2. Word2Vec (CBOW and Skip-gram)
Both models use identical hyperparameters so that only the training scheme differs (sg=0 for CBOW, sg=1 for Skip-gram).
Hyperparameter
Value
vector_size                100
window                      5
min_count                   5
epochs                      5
seed                        42
workers                     4
Resulting vocabulary size   38,984

Model                      Training runtime (s)         Vocabulary size
Word2Vec CBOW (sg=0)           28.20                        38,984
Word2Vec Skip-gram (sg=1)     90.39                         38,984
Both models were trained with gensim.models.Word2Vec in src/embeddings.py. They have the same vocabulary size, because the corpus and min_count are identical: only words that occur at least 5 times are kept (38,984 of the 98,695 unique tokens).
from gensim.models import Word2Vec

PARAMS = dict(vector_size=100, window=5, min_count=5,
              epochs=5, seed=42, workers=4)

# B1: Word2Vec CBOW (sg=0)
cbow = Word2Vec(sentences, sg=0, **PARAMS)

# B2: Word2Vec Skip-gram (sg=1), same hyperparameters
skipgram = Word2Vec(sentences, sg=1, **PARAMS)

B3. CBOW vs Skip-gram
CBOW (Continuous Bag of Words) predicts the centre word from its surrounding context words (context to centre). The context words are averaged into a single vector, so there is only one prediction per window. Skip-gram does the reverse: it predicts each surrounding word from the centre word (centre to context), so every (centre, context) pair becomes a separate training example.
This difference explains the usual trade-off. CBOW trains faster and its averaging works well for frequent words that occur in many contexts. Skip-gram is slower, but a rare word receives many more updates than it would under CBOW, so it tends to represent rare words better. In a movie-review corpus, rare terms such as specific director, genre or technical words are exactly where we expect Skip-gram to do better.
In our runs, CBOW trained in 28.20 seconds and Skip-gram in 90.39 seconds (CBOW was about 3.2 times faster, as expected). For the in-vocabulary probe words (movie, acting, horror, director) the picture is mixed. For the very frequent word “movie”, CBOW gave clean neighbours (film, movies, flick, really, films), while Skip-gram returned rarer and noisier tokens (atleast, sequal, suprise, hellbent). For “acting”, CBOW stayed close to the topic (performances, casting, directing), whereas Skip-gram drifted to unusual words (varies, staggeringly, complimented). For the more specific words Skip-gram looked slightly better: for “horror” both gave sensible genre words (CBOW: slasher, exploitation, zombie, splatter; Skip-gram: slasher, slashers, omnibus, amicus, goriest), and for “director” Skip-gram added related film roles (writer, screenwriter, cinematographer) while CBOW mostly returned variants of the same word (filmmaker, directors, filmmakers). So our results only partly agree with the expectation: Skip-gram was somewhat better on rarer, topic-specific words, but CBOW was cleaner on very frequent words. The corpus has about 6.2 million tokens and the models were trained for only 5 epochs, so this comparison should be interpreted cautiously.
B4. fastText
Word2Vec treats every word as one indivisible unit, so a word that is not in the training vocabulary has no vector. fastText represents each word as a bag of character n-grams (for example, the word “where” contributes pieces such as <wh, whe, her, ere, re>) and builds the word vector from these sub-word pieces. This lets fastText produce a sensible vector for rare, misspelled or unseen words by reusing the pieces it has seen in other words.
from gensim.models import FastText

# B4: fastText (character n-gram sub-words), same hyperparameters
fasttext = FastText(sentences, sg=0, **PARAMS)
fastText was trained with the same hyperparameters as above (training runtime 117.72 s, vocabulary size 38,984).
Out-of-vocabulary / misspelled word demonstration (query words: cinematographyyy, horrrorrr):
Model
cinematographyyy
horrrorrr
Word2Vec (CBOW)
KeyError (word not in vocabulary)
KeyError (word not in vocabulary)
Word2Vec (Skip-gram)  KeyError (word not in vocabulary)   KeyError (word not in vocabulary)
fastText (top-5 similar words)   cinematography (0.9942), cinematograph (0.9930), cinematographic (0.9431), cinematographer (0.9295), cinematographers (0.9180)
horror (0.8951), horrors (0.8709), horrorfest (0.8119), horrid (0.7461), terror (0.7215)
Both Word2Vec models raise a KeyError for the two misspelled words, because these words never appeared in the training vocabulary and so have no vector at all. fastText also does not have them in its vocabulary (in-vocabulary check: False), but it builds a vector from the character n-grams they share with known words, and returns sensible neighbours. For “cinematographyyy” all five neighbours are forms of cinematography (similarity 0.92 to 0.99), and for “horrrorrr” it returns horror, horrors and horrorfest first, followed by horrid and terror. The neighbours for “horrrorrr” are a little less precise (0.72 to 0.90) than for “cinematographyyy”, because the repeated letters change more of the character n-grams, but the intended word is still ranked first. This shows the practical benefit of sub-word information on noisy review text, where misspelled and stretched words are common.

  ```
- **Part C — Numerical Verification & Semantic Probing:**
  ```bash
  python -m src.probing
  ```
- **Part D — Model Packaging & Unified Vectorization Interface:**
  1. Train and save all 4 models (`tfidf.pkl`, `cbow.model`, `skipgram.model`, `fasttext.model`) to `models/` once:
     ```bash
     python -m src.save_models
     ```
  2. Vectorize any raw document using `vectorize(raw_doc, method)` without retraining:
     ```bash
     python -m src.vectorize
     ```
