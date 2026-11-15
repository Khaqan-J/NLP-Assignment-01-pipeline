import time
import pandas as pd
from gensim.models import Word2Vec, FastText
from src.config import CLEANED_CSV_PATH, ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"

def train_and_evaluate_embeddings(save_to_disk=True):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_CSV_PATH).dropna(subset=["text_clean"])
    sentences = [t.split() for t in df["text_clean"].astype(str)]

    params = dict(vector_size=100, window=5, min_count=5, epochs=5, seed=42, workers=4)
    print("=== Part B1 & B2: Hyperparameters ===")
    for k, v in params.items():
        if k != "workers":
            print(f"  {k:15s}: {v}")

    t0 = time.perf_counter()
    cbow_model = Word2Vec(sentences=sentences, sg=0, **params)
    t_cbow = time.perf_counter() - t0
    print(f"\nTrained Word2Vec CBOW (sg=0)      | Vocab size: {len(cbow_model.wv):,} | Time: {t_cbow:.1f}s")

    t0 = time.perf_counter()
    sg_model = Word2Vec(sentences=sentences, sg=1, **params)
    t_sg = time.perf_counter() - t0
    print(f"Trained Word2Vec Skip-gram (sg=1) | Vocab size: {len(sg_model.wv):,} | Time: {t_sg:.1f}s")

    t0 = time.perf_counter()
    ft_model = FastText(sentences=sentences, sg=0, **params)
    t_ft = time.perf_counter() - t0
    print(f"Trained fastText (sub-word ngrams)| Vocab size: {len(ft_model.wv):,} | Time: {t_ft:.1f}s")

    if save_to_disk:
        cbow_model.save(str(MODELS_DIR / "cbow.model"))
        sg_model.save(str(MODELS_DIR / "skipgram.model"))
        ft_model.save(str(MODELS_DIR / "fasttext.model"))

    print("\n=== Part B4: Out-of-Vocabulary (OOV) / Misspelled Word Demonstration ===")
    oov_word = "cinematographyyy"
    print(f"Query OOV / Misspelled Word: '{oov_word}'")
    for name, mdl in [("Word2Vec (CBOW)", cbow_model), ("Word2Vec (Skip-gram)", sg_model)]:
        try:
            _ = mdl.wv.most_similar(oov_word, topn=5)
        except KeyError as e:
            print(f"  {name:22s}: KeyError ({e})")

    ft_sim = ft_model.wv.most_similar(oov_word, topn=5)
    ft_str = ", ".join(f"{w} ({s:.3f})" for w, s in ft_sim)
    print(f"  {'fastText':22s}: {ft_str}")
    return cbow_model, sg_model, ft_model

if __name__ == "__main__":
    train_and_evaluate_embeddings()
