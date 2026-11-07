import json
import time
from src.config import CLEANED_CSV_PATH, STATS_JSON_PATH, PROCESSED_DATA_DIR, REPORTS_DIR
from src.data import load_raw_data
from src.tokenization import regex_tokenize
from src.cleaning import clean_corpus

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(drop_duplicates=True)
    df["doc_id"] = df.index
    raw_tokens = [tok for text in df["review"] for tok in regex_tokenize(text)]
    tokens_before, types_before = len(raw_tokens), len(set(raw_tokens))
    t0 = time.perf_counter()
    cleaned_texts, caps_counts, token_counts = clean_corpus(df["review"].tolist(), batch_size=500)
    elapsed = time.perf_counter() - t0
    df["text_clean"] = cleaned_texts
    df["n_caps_words"] = caps_counts
    df["n_tokens_clean"] = token_counts
    clean_tokens_all = [tok for text in cleaned_texts for tok in text.split()]
    tokens_after, types_after = len(clean_tokens_all), len(set(clean_tokens_all))
    reduction_pct = (types_before - types_after) / types_before * 100
    out_cols = ["doc_id", "text_clean", "label", "sentiment", "n_caps_words", "n_tokens_clean"]
    df[out_cols].to_csv(CLEANED_CSV_PATH, index=False)
    stats = {
        "documents": len(df),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "types_before": types_before,
        "types_after": types_after,
        "vocab_reduction_percent": round(reduction_pct, 2),
        "runtime_seconds": round(elapsed, 2),
    }
    with open(STATS_JSON_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Documents: {len(df):,} | Tokens: {tokens_before:,} -> {tokens_after:,} | Types: {types_before:,} -> {types_after:,} (-{reduction_pct:.1f}%)")

if __name__ == "__main__":
    main()
