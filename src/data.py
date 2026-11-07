import urllib.request
import pandas as pd
from src.config import DATA_URL, RAW_CSV_PATH, RAW_DATA_DIR

def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV_PATH.exists():
        print(f"Downloading IMDB 50K dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, RAW_CSV_PATH)
        print(f"Saved raw dataset to {RAW_CSV_PATH}")
    else:
        print(f"Dataset already exists at {RAW_CSV_PATH}")

def load_raw_data(drop_duplicates=False):
    if not RAW_CSV_PATH.exists():
        download_data()
    df = pd.read_csv(RAW_CSV_PATH)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df_raw = load_raw_data(drop_duplicates=False)
    total_rows = len(df_raw)
    missing = df_raw.isnull().sum().to_dict()
    label_counts = df_raw["sentiment"].value_counts().to_dict()

    df_dedup = load_raw_data(drop_duplicates=True)
    dedup_rows = len(df_dedup)
    dropped = total_rows - dedup_rows

    print("=== IMDB 50K Dataset Summary ===")
    print(f"Total reviews (raw)      : {total_rows:,}")
    print(f"Class balance (raw)      : {label_counts}")
    print(f"Missing values           : {missing}")
    print(f"Exact duplicates dropped : {dropped:,}")
    print(f"Reviews after deduplic.  : {dedup_rows:,}")