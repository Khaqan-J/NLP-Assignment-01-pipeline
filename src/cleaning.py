import html
import re
import unicodedata
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.regex_patterns import HTML_TAG, URL, ELONGATION, ALLCAPS

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

SENTIMENT_KEEP = {
    "no", "nor", "not", "don", "don't", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
    "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "never",
    "very", "too", "only", "more", "most", "few", "but", "against"
}
CUSTOM_STOPWORDS = set(stopwords.words("english")) - SENTIMENT_KEEP
QUOTE_DASH_MAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})

def pre_clean_string(text: str):
    text = unicodedata.normalize("NFC", text).translate(QUOTE_DASH_MAP)
    text = html.unescape(text)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    n_caps = len(ALLCAPS.findall(text))
    text = text.lower()
    text = ELONGATION.sub(r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, n_caps

def process_doc_tokens(doc):
    clean_tokens = []
    for tok in doc:
        lemma = tok.lemma_.lower().strip()
        if lemma in ("n't", "not"):
            lemma = "not"
        if not lemma.isalpha() or len(lemma) <= 1:
            continue
        if lemma in CUSTOM_STOPWORDS:
            continue
        clean_tokens.append(lemma)
    return clean_tokens

def clean_text(text: str) -> str:
    pre_text, _ = pre_clean_string(text)
    doc = nlp(pre_text)
    return " ".join(process_doc_tokens(doc))

def clean_corpus(texts, batch_size=500):
    pre_texts, caps_counts = [], []
    for t in texts:
        pt, nc = pre_clean_string(t)
        pre_texts.append(pt)
        caps_counts.append(nc)
    cleaned_texts, token_counts = [], []
    for doc in nlp.pipe(pre_texts, batch_size=batch_size):
        toks = process_doc_tokens(doc)
        cleaned_texts.append(" ".join(toks))
        token_counts.append(len(toks))
    return cleaned_texts, caps_counts, token_counts

def demo_cleaning():
    check_1 = "This movie is not good, and I don't recommend it. Very boring but short."
    check_2 = "It was “GREAT”!!! Sooooo good… but I didn’t love the ending.<br /><br />Visit www.imdb.com &amp; see 9/10. Actors were running & played better than expected."
    print("=== Section 4: Cleaning Pipeline Worked Examples ===")
    print("Input 1 :", check_1)
    print("Output 1:", clean_text(check_1))
    print("\nInput 2 :", check_2)
    print("Output 2:", clean_text(check_2))
    ps = PorterStemmer()
    words = ["movies", "studies", "characters", "happily", "amazingly", "worst"]
    print("\n=== Stemming vs. Lemmatization ===")
    for w in words:
        print(f"{w:12s} | Porter: {ps.stem(w):10s} | spaCy Lemma: {nlp(w)[0].lemma_}")

if __name__ == "__main__":
    demo_cleaning()
