# File: src/explore.py (Alisha - Member B)
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR
from src.data import load_raw_data

def run_exploration():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_data(drop_duplicates=False)
    df_dedup = load_raw_data(drop_duplicates=True)

    # 1. Word length statistics (whitespace split)
    word_counts = df_raw["review"].apply(lambda x: len(x.split()))
    print("=== Section 2: Corpus Exploration Statistics ===")
    print(f"Documents                : {len(df_raw):,} (no missing values, {len(df_raw) - len(df_dedup)} exact duplicates)")
    print(f"Words per review         : mean {word_counts.mean():.1f}, median {word_counts.median():.0f}, min {word_counts.min()}, max {word_counts.max():,}")
    print(f"Target distribution      : {(df_raw['sentiment']=='positive').sum():,} positive, {(df_raw['sentiment']=='negative').sum():,} negative")

    # 2. Sample 5,000 reviews for token/type/hapax stats
    sample_df = df_raw.sample(n=5000, random_state=42)
    token_pattern = re.compile(r"\w+|[^\w\s]")
    all_tokens = []
    for text in sample_df["review"]:
        all_tokens.extend(token_pattern.findall(text))

    counts = Counter(all_tokens)
    n_tokens = len(all_tokens)
    n_types = len(counts)
    ttr = n_types / n_tokens
    hapax = sum(1 for k, v in counts.items() if v == 1)
    top_100_share = sum(v for _, v in counts.most_common(100)) / n_tokens * 100

    print(f"Tokens (5,000 sample)    : {n_tokens:,}")
    print(f"Types (5,000 sample)     : {n_types:,} (TTR: {ttr:.3f})")
    print(f"Hapax legomena (once)    : {hapax:,} ({hapax/n_types*100:.1f}% of types)")
    print(f"Top 100 types coverage   : {top_100_share:.1f}%\n")

    # 3. Mess counts over all 50,000 reviews
    reviews = df_raw["review"]
    mess_checks = {
        "HTML line breaks (<br />)": reviews.str.contains(r"<br\s*/?>", case=False, regex=True),
        "ALL-CAPS words of 4+ letters": reviews.str.contains(r"\b[A-Z]{4,}\b", regex=True),
        "Explicit rating (8/10, 3 out of 10)": reviews.str.contains(r"\b\d+(?:\.\d+)?\s*(?:/\s*10|out of 10)\b", case=False, regex=True),
        "Non-ASCII characters": reviews.str.contains(r"[^\x00-\x7F]", regex=True),
        "Extra spaces or space before punct": reviews.str.contains(r"  +|\s+[,.!?]", regex=True),
        "Elongated words": reviews.str.contains(r"([A-Za-z])\1{2,}", regex=True),
        "URLs": reviews.str.contains(r"https?://|www\.", case=False, regex=True),
    }

    print("=== How Common is the Mess? (50,000 reviews) ===")
    for name, mask in mess_checks.items():
        c = int(mask.sum())
        print(f"{name:38s}: {c:6,d} ({c / len(df_raw) * 100:5.1f}%)")

    # 4. Real examples from the data
    print("\n=== Real Messy Examples ===")
    for row_idx, label in [(0, "HTML"), (3, "ALL-CAPS & Rating"), (23, "Elongation"), (25, "Spacing"), (29, "Non-ASCII")]:
        snippet = df_raw.loc[row_idx, "review"][:180].replace("\n", " ")
        print(f"{label} (row {row_idx}): {snippet}...")

    # 5. Save Plot 1: Zipf log-log plot
    freqs = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)
    plt.figure(figsize=(7, 5))
    plt.loglog(ranks, freqs, color="navy")
    plt.title("Zipf's Law on Raw IMDB Sample (Log-Log)")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "zipf_raw.png", dpi=150)
    plt.close()

    # 6. Save Plot 2: Length and Labels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(word_counts, bins=50, color="steelblue", edgecolor="black")
    axes[0].set_title("Review Length Distribution (Words)")
    axes[0].set_xlabel("Words per Review")
    axes[0].set_ylabel("Count")
    df_raw["sentiment"].value_counts().plot(kind="bar", ax=axes[1], color=["seagreen", "indianred"])
    axes[1].set_title("Target Label Distribution")
    axes[1].set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "length_and_labels.png", dpi=150)
    plt.close()
    print(f"\nSaved plots to {FIGURES_DIR}/zipf_raw.png and {FIGURES_DIR}/length_and_labels.png")

if __name__ == "__main__":
    run_exploration()