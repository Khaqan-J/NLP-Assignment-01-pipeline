import nltk

def download_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        print(f"Downloading NLTK resource: {res}...")
        nltk.download(res, quiet=True)
    print("All NLTK resources ready.")

if __name__ == "__main__":
    download_nltk_resources()