from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset URL & Local Paths
DATA_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
RAW_CSV_PATH = RAW_DATA_DIR / "IMDB_Dataset.csv"
CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "imdb_cleaned.csv"
STATS_JSON_PATH = REPORTS_DIR / "cleaning_stats.json"

RANDOM_SEED = 42