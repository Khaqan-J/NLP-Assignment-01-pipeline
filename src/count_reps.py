import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.config import CLEANED_CSV_PATH

def run_count_representations():
    df = pd.read_csv(CLEANED_CSV_PATH).dropna(subset=["text_clean"])
    texts = df["text_clean"].astype(str).tolist()
    n_docs = len(texts)
    token_lists = [t.split() for t in texts]
    total_tokens = sum(len(t) for t in token_lists)
    unique_vocab = len(set(w for t in token_lists for w in t))
    avg_len = total_tokens / n_docs

    print("=== Section 1: Cleaned Corpus Summary ===")
    print(f"Number of documents              : {n_docs:,}")
    print(f"Total tokens                     : {total_tokens:,}")
    print(f"Vocabulary size (unique tokens)  : {unique_vocab:,}")
    print(f"Average document length (tokens) : {avg_len:.1f}\n")

    print("=== Part A1: Bag of Words (CountVectorizer) ===")
    bow_vec = CountVectorizer()
    X_bow = bow_vec.fit_transform(texts)
    rows, cols = X_bow.shape
    nnz = X_bow.nnz
    sparsity = 100.0 * (1.0 - nnz / (rows * cols))
    print(f"Matrix shape (documents x vocab) : ({rows:,}, {cols:,})")
    print(f"Non-zero entries                 : {nnz:,}")
    print(f"Sparsity (% of zero entries)     : {sparsity:.4f}%\n")

    print("=== Part A2: TF-IDF (High-Weight vs. Near-Zero Common Words) ===")
    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(texts)
    vocab = np.array(tfidf_vec.get_feature_names_out())
    doc_freqs = np.bincount(X_tfidf.indices, minlength=len(vocab))

    doc_idx = 0
    print(f"Example Document Index: {doc_idx}")
    print(f"Excerpt: {texts[doc_idx][:140]}...")
    row_vec = X_tfidf[doc_idx].toarray().ravel()
    nz_indices = np.where(row_vec > 0)[0]
    sorted_nz = nz_indices[np.argsort(row_vec[nz_indices])[::-1]]

    print("\nTop 3 High-Weight Words in Document 0:")
    for idx in sorted_nz[:3]:
        print(f"  High weight      | Word: {vocab[idx]:15s} | TF-IDF: {row_vec[idx]:.4f} | Doc Freq (df): {doc_freqs[idx]:,}")

    print("\nBottom 3 Lowest-Weight (Common) Words Present in Document 0:")
    for idx in sorted_nz[-3:]:
        print(f"  Near zero/common | Word: {vocab[idx]:15s} | TF-IDF: {row_vec[idx]:.4f} | Doc Freq (df): {doc_freqs[idx]:,}")

    print("\n=== Part A3: N-gram Growth & Sparsity Table (min_df=5) ===")
    for n_range, label in [((1, 1), "Unigrams (1,1)"), ((1, 2), "Uni + bigrams (1,2)"), ((1, 3), "Uni + bi + trigrams (1,3)")]:
        vec = TfidfVectorizer(ngram_range=n_range, min_df=5)
        X_ng = vec.fit_transform(texts)
        r, c = X_ng.shape
        sp = 100.0 * (1.0 - X_ng.nnz / (r * c))
        print(f"{label:26s} | Features: {c:9,d} | Shape: ({r:,}, {c:,}) | Sparsity: {sp:.4f}%")

if __name__ == "__main__":
    run_count_representations()
