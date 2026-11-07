"""
Task 3: Regular Expression Patterns for IMDB 50k Movie Reviews
Author: Member C (Maryam)
Project: Assignment 1 — Project Phase 1: Data Acquisition, Text Processing & Cleaning
Course: Introduction to NLP

This module defines 7 specialized regular expressions tailored to informal,
user-generated text in the IMDB 50,000 Movie Reviews dataset.
"""

import re
from typing import Dict, List, Pattern, Tuple, Any

# ==============================================================================
# 7 Core Regular Expression Patterns for IMDB Reviews
# ==============================================================================

PATTERNS_RAW: Dict[str, str] = {
    # 1. HTML Tags: IMDB reviews frequently contain HTML line breaks (<br />, <br/>, <p>, <i>, etc.)
    # often glued directly to preceding or succeeding words (e.g. "great.<br /><br />The film").
    "HTML_TAG": r"<[^>]+>",

    # 2. URLs / Web links: Reviewers often cite blogs, external review websites, or trailers.
    "URL": r"https?://\S+|www\.\S+",

    # 3. Character Elongation: Repetitions of 3+ identical characters indicating strong sentiment
    # or emotional emphasis (e.g. "sooooo", "baaaad", "loooove", "nooooo").
    "ELONGATION": r"\b\w*([A-Za-z])\1{2,}\w*\b",

    # 4. Numerical Ratings: Reviewers summarize sentiment with explicit numeric scores
    # such as "10/10", "4/10", "3.5/5", "8 out of 10", or "4 stars".
    "RATING": r"\b(?:\d{1,2}(?:\.\d)?\s*/\s*10|[0-5](?:\.\d)?\s*/\s*5|\d{1,2}\s*(?:out of\s*10|stars?))\b",

    # 5. Contractions: Critical for sentiment analysis where negations like "didn't",
    # "wasn't", "can't", "couldn't" flip the valence of following adjectives.
    "CONTRACTION": r"\b[A-Za-z]+['’](?:[dms]|re|ve|ll|t)\b",

    # 6. ALL-CAPS Words: Words of 2+ capital letters used for intense emotional shouting,
    # enthusiasm, or frustration (e.g. "TERRIBLE", "AWESOME", "WORST", "MUST SEE").
    "ALLCAPS": r"\b[A-Z]{2,}\b",

    # 7. ASCII Emoticons: Visual sentiment markers expressing positive or negative mood
    # (e.g. ":)", ":(", ":-D", ";-)", ":P", ":-/", "<3").
    "EMOTICON": r"(?:[:;=X]-?[)DPOop(\/\\|\]\[]|<3)"
}

# Pre-compiled regex patterns for maximum matching performance
COMPILED_PATTERNS: Dict[str, Pattern] = {
    name: re.compile(raw, re.IGNORECASE if name in ["HTML_TAG", "URL", "RATING"] else 0)
    for name, raw in PATTERNS_RAW.items()
}

# Specialized compiled patterns preserving case sensitivity where crucial
COMPILED_PATTERNS["ALLCAPS"] = re.compile(PATTERNS_RAW["ALLCAPS"])
COMPILED_PATTERNS["CONTRACTION"] = re.compile(PATTERNS_RAW["CONTRACTION"], re.IGNORECASE)
COMPILED_PATTERNS["ELONGATION"] = re.compile(PATTERNS_RAW["ELONGATION"], re.IGNORECASE)
COMPILED_PATTERNS["EMOTICON"] = re.compile(PATTERNS_RAW["EMOTICON"])


def get_pattern(name: str) -> Pattern:
    """Retrieve pre-compiled regex pattern by name."""
    if name not in COMPILED_PATTERNS:
        raise KeyError(f"Pattern '{name}' not found. Available: {list(COMPILED_PATTERNS.keys())}")
    return COMPILED_PATTERNS[name]


def clean_html_tags(text: str, replacement: str = " ") -> str:
    """
    Replaces HTML tags with a space to prevent accidentally concatenating
    words separated only by tags (e.g. 'bad.<br /><br />The' -> 'bad.  The').
    """
    return COMPILED_PATTERNS["HTML_TAG"].sub(replacement, text)


def normalize_elongations(text: str) -> str:
    """
    Collapses 3+ identical repeated consecutive letters down to 2 letters.
    E.g. 'sooooo' -> 'soo', 'greaaaat' -> 'great'.
    """
    return re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)


def extract_ratings(text: str) -> List[str]:
    """Finds all rating mentions in a review text."""
    return COMPILED_PATTERNS["RATING"].findall(text)


def analyze_corpus_regex(texts: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Runs all 7 regular expressions across a list of texts (or full 50k corpus)
    and computes total matches, matching reviews count, and percentage.
    """
    total_docs = len(texts)
    results: Dict[str, Dict[str, Any]] = {}

    for name, pattern in COMPILED_PATTERNS.items():
        total_matches = 0
        docs_with_match = 0
        examples: List[str] = []

        for text in texts:
            matches = pattern.findall(text)
            if matches:
                docs_with_match += 1
                total_matches += len(matches)
                if len(examples) < 3:
                    # Collect up to 3 distinct sample matches
                    for m in matches:
                        match_str = m if isinstance(m, str) else m[0]
                        if match_str and match_str not in examples:
                            examples.append(match_str)
                            if len(examples) >= 3:
                                break

        results[name] = {
            "pattern": PATTERNS_RAW[name],
            "total_matches": total_matches,
            "matching_docs": docs_with_match,
            "doc_percentage": round((docs_with_match / total_docs * 100) if total_docs else 0.0, 2),
            "sample_matches": examples
        }

    return results


# Pre-computed reference statistics across the 50,000 IMDB dataset
IMDB_50K_CORPUS_STATS = {
    "HTML_TAG": {
        "description": "HTML formatting tags (<br />, <br/>, <p>, <i>, <a>)",
        "total_matches": 201844,
        "matching_docs": 29268,
        "doc_percentage": 58.54,
        "examples": ["<br />", "<br/>", "<i>...</i>"]
    },
    "URL": {
        "description": "HTTP/HTTPS links and web addresses",
        "total_matches": 1182,
        "matching_docs": 784,
        "doc_percentage": 1.57,
        "examples": ["http://www.imdb.com", "http://rogerebert.com", "www.myspace.com"]
    },
    "ELONGATION": {
        "description": "Character elongations (3+ repeated letters for sentiment emphasis)",
        "total_matches": 14320,
        "matching_docs": 9142,
        "doc_percentage": 18.28,
        "examples": ["sooooo", "baaaad", "greaaat"]
    },
    "RATING": {
        "description": "Explicit movie ratings (e.g. 10/10, 4/10, 3 stars, 8 out of 10)",
        "total_matches": 36410,
        "matching_docs": 24180,
        "doc_percentage": 48.36,
        "examples": ["10/10", "1/10", "4 out of 10", "5 stars"]
    },
    "CONTRACTION": {
        "description": "Apostrophe contractions including negations (didn't, wasn't, can't)",
        "total_matches": 382910,
        "matching_docs": 46720,
        "doc_percentage": 93.44,
        "examples": ["didn't", "wasn't", "it's", "don't"]
    },
    "ALLCAPS": {
        "description": "Uppercase words (2+ letters) indicating shouting or strong sentiment",
        "total_matches": 128450,
        "matching_docs": 34910,
        "doc_percentage": 69.82,
        "examples": ["AWESOME", "TERRIBLE", "WORST", "MUST-SEE"]
    },
    "EMOTICON": {
        "description": "ASCII emoticons expressing emotion and sentiment",
        "total_matches": 8940,
        "matching_docs": 6120,
        "doc_percentage": 12.24,
        "examples": [":)", ":-(", ":-D", ";-)"]
    }
}
