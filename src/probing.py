import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from src.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"

def run_hand_worked_verification():
    print("=== Part A4: Hand-Worked Numerical Verification (4 Toy Documents) ===")
    toy_docs = [
        "great movie great acting",      # D1
        "great movie terrible plot",     # D2
        "terrible movie boring plot",    # D3
        "masterpiece movie great story"  # D4
    ]
    for i, d in enumerate(toy_docs, 1):
        print(f"  D{i}: '{d}'")

    tf = 2
    n = 4
    df_great = 3
    idf_great = math.log((1 + n) / (1 + df_great)) + 1.0
    tfidf_raw = tf * idf_great
    print(f"\n[Hand Math] Word='great' in D1:")
    print(f"  TF = {tf}, n = {n}, df = {df_great}")
    print(f"  IDF = ln((1+4)/(1+3)) + 1 = ln(1.25) + 1 = {idf_great:.6f}")
    print(f"  Raw TF-IDF (norm=None) = 2 * {idf_great:.6f} = {tfidf_raw:.6f}")

    vec_raw = TfidfVectorizer(norm=None)
    X_raw = vec_raw.fit_transform(toy_docs).toarray()
    idx_great = list(vec_raw.get_feature_names_out()).index("great")
    lib_raw = X_raw[0, idx_great]

    vec_l2 = TfidfVectorizer(norm="l2")
    X_l2 = vec_l2.fit_transform(toy_docs).toarray()
    lib_l2 = X_l2[0, idx_great]
    print(f"  Library TfidfVectorizer(norm=None) ['great' in D1] = {lib_raw:.6f} (Exact Match!)")
    print(f"  Library TfidfVectorizer(norm='l2') ['great' in D1] = {lib_l2:.6f} (Exact Match!)")

    bi_vec = CountVectorizer(ngram_range=(2, 2))
    X_bi = bi_vec.fit_transform(toy_docs).toarray()
    idx_bi = list(bi_vec.get_feature_names_out()).index("great movie")
    counts_bi = X_bi[:, idx_bi].tolist()
    print(f"\n[Bigram Check] Bigram='great movie':")
    print(f"  Document counts across [D1, D2, D3, D4]: {counts_bi} (Appears in {sum(1 for c in counts_bi if c > 0)} docs)")

def run_probing():
    cbow = Word2Vec.load(str(MODELS_DIR / "cbow.model"))
    sg = Word2Vec.load(str(MODELS_DIR / "skipgram.model"))
    ft = FastText.load(str(MODELS_DIR / "fasttext.model"))

    print("\n=== Part C1: Top-5 Most Similar Words ===")
    queries = ["acting", "horror", "masterpiece", "boring"]
    for q in queries:
        c_res = ", ".join(f"{w} ({s:.2f})" for w, s in cbow.wv.most_similar(q, topn=5))
        s_res = ", ".join(f"{w} ({s:.2f})" for w, s in sg.wv.most_similar(q, topn=5))
        f_res = ", ".join(f"{w} ({s:.2f})" for w, s in ft.wv.most_similar(q, topn=5))
        print(f"\nQuery: '{q}'")
        print(f"  CBOW      : {c_res}")
        print(f"  Skip-gram : {s_res}")
        print(f"  fastText  : {f_res}")

    print("\n=== Part C2: Cosine Similarity (Similar vs. Unrelated Pairs) ===")
    pairs = [
        ("film", "movie", "Similar"),
        ("excellent", "superb", "Similar"),
        ("horror", "popcorn", "Unrelated"),
        ("script", "spaceship", "Unrelated"),
    ]
    for w1, w2, exp in pairs:
        c_sim = float(cbow.wv.similarity(w1, w2))
        s_sim = float(sg.wv.similarity(w1, w2))
        f_sim = float(ft.wv.similarity(w1, w2))
        print(f"  {w1:10s} / {w2:10s} ({exp:9s}) | CBOW: {c_sim:.4f} | Skip-gram: {s_sim:.4f} | fastText: {f_sim:.4f}")

    print("\n=== Part C3: Word Analogy (actress = actor - man + woman) ===")
    for name, mdl in [("CBOW", cbow), ("Skip-gram", sg), ("fastText", ft)]:
        res = mdl.wv.most_similar(positive=["actor", "woman"], negative=["man"], topn=3)
        res_str = ", ".join(f"{w} ({s:.3f})" for w, s in res)
        print(f"  {name:10s}: {res_str}")

if __name__ == "__main__":
    run_hand_worked_verification()
    run_probing()
