"""Part B: learned word embeddings (Word2Vec CBOW, Word2Vec Skip-gram, fastText)
and an out-of-vocabulary (OOV) demonstration.

Run from the repository root:
    python -m src.embeddings

All three models share identical hyperparameters so the comparison is fair;
only the training scheme differs (CBOW: sg=0, Skip-gram: sg=1, fastText: sub-words).
"""

import time
from pathlib import Path

import pandas as pd
from gensim.models import FastText, Word2Vec

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "imdb_cleaned.csv"
MODEL_DIR = ROOT / "models"  # saved locally only, do NOT commit .model files

PARAMS = dict(
    vector_size=100,
    window=5,
    min_count=5,
    epochs=5,
    seed=42,
    workers=4,
)

# Misspelled / unseen domain words used for the OOV demonstration
OOV_WORDS = ["cinematographyyy", "horrrorrr"]


def load_sentences(path=DATA_PATH):
    """Return the cleaned reviews as a list of token lists."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first with: python -m src.run_pipeline"
        )
    df = pd.read_csv(path)
    return [str(text).split() for text in df["text_clean"].dropna()]


def _timed(train_fn):
    """Run train_fn() and return (model, seconds)."""
    start = time.perf_counter()
    model = train_fn()
    return model, time.perf_counter() - start


def train_cbow(sentences):
    """B1: Word2Vec CBOW (sg=0)."""
    return _timed(lambda: Word2Vec(sentences, sg=0, **PARAMS))


def train_skipgram(sentences):
    """B2: Word2Vec Skip-gram (sg=1), same hyperparameters as CBOW."""
    return _timed(lambda: Word2Vec(sentences, sg=1, **PARAMS))


def train_fasttext(sentences):
    """B4: fastText (character n-gram sub-words), same hyperparameters."""
    return _timed(lambda: FastText(sentences, sg=0, **PARAMS))


def describe(name, model, seconds):
    print(f"\n=== {name} ===")
    print("Hyperparameters :", PARAMS)
    print(f"Training runtime: {seconds:.2f} s")
    print(f"Vocabulary size : {len(model.wv)}")


def oov_demo(cbow, skipgram, fasttext, words=OOV_WORDS):
    """Show that Word2Vec fails on OOV words while fastText still returns vectors."""
    for word in words:
        print(f"\n--- OOV / misspelled word: '{word}' ---")
        for name, model in (("Word2Vec CBOW", cbow), ("Word2Vec Skip-gram", skipgram)):
            try:
                model.wv[word]
                print(f"{name}: found in vocabulary (unexpected, try another word)")
            except KeyError:
                print(f"{name}: KeyError (word not in vocabulary)")
        print("fastText in vocabulary:", word in fasttext.wv.key_to_index)
        print("fastText top-5 similar words (via character n-grams):")
        for neighbour, score in fasttext.wv.most_similar(word, topn=5):
            print(f"   {neighbour:<20} {score:.4f}")


def save_models(cbow, skipgram, fasttext, model_dir=MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    cbow.save(str(model_dir / "cbow.model"))
    skipgram.save(str(model_dir / "skipgram.model"))
    fasttext.save(str(model_dir / "fasttext.model"))
    print(f"\nModels saved in {model_dir} (not committed to git).")


def main():
    sentences = load_sentences()
    n_tokens = sum(len(s) for s in sentences)
    print(f"Loaded {len(sentences)} documents, {n_tokens} tokens")

    cbow, t_cbow = train_cbow(sentences)
    describe("B1: Word2Vec CBOW", cbow, t_cbow)

    skipgram, t_sg = train_skipgram(sentences)
    describe("B2: Word2Vec Skip-gram", skipgram, t_sg)

    fasttext, t_ft = train_fasttext(sentences)
    describe("B4: fastText", fasttext, t_ft)

    oov_demo(cbow, skipgram, fasttext)
    save_models(cbow, skipgram, fasttext)


if __name__ == "__main__":
    main()
