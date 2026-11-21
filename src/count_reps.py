
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

# Dataset location
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "imdb_cleaned.csv"
)


# ==========================================
# LOAD CLEANED DATASET
# ==========================================

def load_texts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "text_clean" not in df.columns:
        raise ValueError(
            "Column 'text_clean' not found!"
        )

    texts = df["text_clean"].dropna().astype(str).tolist()

    if not texts:
        raise ValueError("No reviews found!")

    return texts


# ==========================================
# SECTION 1: CORPUS SUMMARY
# ==========================================

def corpus_summary(texts):

    token_lists = [text.split() for text in texts]

    total_tokens = sum(len(tokens) for tokens in token_lists)

    vocabulary = set(
        word
        for tokens in token_lists
        for word in tokens
    )

    avg_length = total_tokens / len(texts)

    print("\n===== SECTION 1: CORPUS SUMMARY =====")

    print("Number of documents:", len(texts))
    print("Total tokens:", total_tokens)
    print("Vocabulary size:", len(vocabulary))
    print("Average document length:", round(avg_length, 2))

    return {
        "documents": len(texts),
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocabulary),
        "average_document_length": avg_length
    }


# ==========================================
# PART A1: BAG OF WORDS
# ==========================================

def bow_analysis(texts):

    print("\n===== PART A1: BAG OF WORDS =====")

    vectorizer = CountVectorizer()

    X_bow = vectorizer.fit_transform(texts)

    rows, cols = X_bow.shape

    non_zero = X_bow.nnz

    sparsity = 100 * (
        1 - non_zero / (rows * cols)
    )

    print("Matrix shape:", X_bow.shape)
    print("Non-zero entries:", non_zero)
    print(f"Sparsity: {sparsity:.4f}%")

    return vectorizer, X_bow


# ==========================================
# PART A2: TF-IDF
# ==========================================

def tfidf_analysis(texts):

    print("\n===== PART A2: TF-IDF =====")

    vectorizer = TfidfVectorizer()

    X_tfidf = vectorizer.fit_transform(texts)

    features = vectorizer.get_feature_names_out()

    # Document frequency
    df = np.asarray(
        (X_tfidf > 0).sum(axis=0)
    ).ravel()

    # First document
    row = X_tfidf.getrow(0)

    weights = row.data
    indices = row.indices

    order = np.argsort(weights)

    low = order[:3]
    high = order[-3:][::-1]

    print("\nDocument 0 excerpt:")
    print(texts[0][:160])

    print("\nType          | Word | TF-IDF Weight | Document Frequency")

    for label, selected in [
        ("High weight", high),
        ("Near zero", low)
    ]:

        for pos in selected:

            feature_index = indices[pos]

            print(
                f"{label:13} | "
                f"{features[feature_index]:15} | "
                f"{weights[pos]:.4f} | "
                f"{df[feature_index]}"
            )

    return vectorizer, X_tfidf


# ==========================================
# PART A3: N-GRAM EXPERIMENTS
# ==========================================

def ngram_analysis(texts):

    print("\n===== PART A3: N-GRAM ANALYSIS =====")

    print("Using min_df=5 for all configurations.")

    print("\nN-gram Range | Features | Matrix Shape | Sparsity")

    results = []

    for ngram in [(1, 1), (1, 2), (1, 3)]:

        vectorizer = CountVectorizer(
            ngram_range=ngram,
            min_df=5
        )

        X = vectorizer.fit_transform(texts)

        rows, cols = X.shape

        sparsity = 100 * (
            1 - X.nnz / (rows * cols)
        )

        results.append({
            "N-gram range": str(ngram),
            "Features": cols,
            "Matrix shape": X.shape,
            "Sparsity (%)": round(sparsity, 4)
        })

        print(
            f"{str(ngram):12} | "
            f"{cols:8} | "
            f"{X.shape} | "
            f"{sparsity:.4f}%"
        )

    return results


# ==========================================
# RUN ALL EXPERIMENTS
# ==========================================

def run_count_representations():

    texts = load_texts()

    corpus_summary(texts)

    bow_analysis(texts)

    tfidf_analysis(texts)

    ngram_analysis(texts)


if __name__ == "__main__":
    run_count_representations()
