import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import CLEANED_CSV_PATH, ROOT_DIR
from src.embeddings import train_and_evaluate_embeddings

MODELS_DIR = ROOT_DIR / "models"

def save_all_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_CSV_PATH).dropna(subset=["text_clean"])
    texts = df["text_clean"].astype(str).tolist()

    print("Fitting and saving TF-IDF vectorizer to models/tfidf.pkl...")
    tfidf = TfidfVectorizer(max_features=50000)
    tfidf.fit(texts)
    joblib.dump(tfidf, MODELS_DIR / "tfidf.pkl")
    print(f"Saved models/tfidf.pkl (vocab size: {len(tfidf.vocabulary_):,})")

    print("Training and saving CBOW, Skip-gram, and fastText models to models/...")
    train_and_evaluate_embeddings(save_to_disk=True)
    print("All 4 models successfully saved in models/!")

if __name__ == "__main__":
    save_all_models()
