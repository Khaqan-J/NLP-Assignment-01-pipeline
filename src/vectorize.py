import html
import re
import joblib
import numpy as np
from gensim.models import Word2Vec, FastText
from nltk.corpus import stopwords
from src.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
_CACHE = {}

SENTIMENT_KEEP = {
    "no", "nor", "not", "don", "don't", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "never",
    "very", "too", "only", "more", "most", "few", "but", "against"
}

def _clean_doc_for_vectorize(raw_doc: str) -> str:
    try:
        from src.cleaning import clean_text
        return clean_text(raw_doc)
    except Exception:
        stops = set(stopwords.words("english")) - SENTIMENT_KEEP
        t = html.unescape(str(raw_doc))
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"(?:https?://|www\.)\S+", " ", t, flags=re.I)
        t = t.lower().replace("n't", " not ").replace("'", " ")
        t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)
        toks = [w for w in re.findall(r"[a-z]+", t) if len(w) > 1 and w not in stops]
        return " ".join(toks)

def _load_model(method: str):
    method = method.lower().strip()
    if method not in _CACHE:
        if method == "tfidf":
            _CACHE[method] = joblib.load(MODELS_DIR / "tfidf.pkl")
        elif method == "cbow":
            _CACHE[method] = Word2Vec.load(str(MODELS_DIR / "cbow.model"))
        elif method == "skipgram":
            _CACHE[method] = Word2Vec.load(str(MODELS_DIR / "skipgram.model"))
        elif method == "fasttext":
            _CACHE[method] = FastText.load(str(MODELS_DIR / "fasttext.model"))
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: tfidf, cbow, skipgram, fasttext.")
    return _CACHE[method]

def vectorize(raw_doc: str, method: str = "tfidf") -> np.ndarray:
    """Clean a raw document using Phase 1 pipeline and return its vector under the chosen representation."""
    cleaned = _clean_doc_for_vectorize(raw_doc)
    tokens = cleaned.split()
    model = _load_model(method)

    if method.lower() == "tfidf":
        return model.transform([cleaned]).toarray()[0]

    wv = model.wv
    valid_vecs = [wv[w] for w in tokens if w in wv]
    if not valid_vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean(valid_vecs, axis=0)

if __name__ == "__main__":
    sample_doc = "This movie had breathtaking cinematography and brilliant acting, truly a masterpiece!"
    print(f"Sample Raw Document: {sample_doc}\n")
    for m in ["tfidf", "cbow", "skipgram", "fasttext"]:
        v = vectorize(sample_doc, method=m)
        print(f"Method: {m:10s} | Vector shape: {v.shape} | L2 Norm: {np.linalg.norm(v):.4f} | First 5 dims: {np.round(v[:5], 4)}")
