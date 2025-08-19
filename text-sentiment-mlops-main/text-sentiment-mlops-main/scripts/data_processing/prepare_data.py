import os
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"




def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("<br />", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = ' '.join(text.split())
    return text

def download_amazon_reviews():
    print("Downloading Amazon reviews from Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("kritanjalijain/amazon-reviews", path=str(RAW_DATA_DIR), unzip=True)

def load_and_prepare_imdb():
    print("Preparing IMDb dataset...")
    imdb = load_dataset("imdb")
    train_df = pd.DataFrame(imdb["train"])
    test_df = pd.DataFrame(imdb["test"])
    for df in [train_df, test_df]:
        df['sentiment'] = df['label'].map({1: "positive", 0: "negative"})
        df['text'] = df['text'].apply(preprocess_text)
        df['source'] = "imdb"
    return pd.concat([train_df[['text', 'sentiment', 'source']], test_df[['text', 'sentiment', 'source']]])

def load_and_prepare_amazon():
    print("Preparing Amazon dataset from Kaggle...")

    file_path = RAW_DATA_DIR / "train.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Amazon reviews train.csv not found at: {file_path}")

    # Dataset has no header row; assign manually
    df = pd.read_csv(file_path, header=None)
    df.columns = ['label', 'title', 'text']

    df['sentiment'] = df['label'].apply(lambda x: "negative" if x == 1 else "positive")
    df['text'] = df['text'].apply(preprocess_text)
    df['source'] = "amazon"

    return df[['text', 'sentiment', 'source']]

def load_and_prepare_tweets():
    print("Preparing Sentiment140 tweet dataset from Hugging Face...")
    sentiment140 = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)
    df = pd.DataFrame(sentiment140["train"])
    df = df.rename(columns={"text": "text", "sentiment": "label"})
    df['sentiment'] = df['label'].map({0: "negative", 4: "positive"})
    df = df.dropna(subset=['sentiment'])
    df['text'] = df['text'].apply(preprocess_text)
    df['source'] = "sentiment140"
    return df[['text', 'sentiment', 'source']]

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    imdb_df = load_and_prepare_imdb()
    amazon_df = load_and_prepare_amazon()
    tweet_df = load_and_prepare_tweets()

    all_data = pd.concat([imdb_df, amazon_df, tweet_df], ignore_index=True)
    all_data = all_data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_df, test_df = train_test_split(all_data, test_size=0.2, stratify=all_data["sentiment"], random_state=42)

    train_df.to_csv(PROCESSED_DATA_DIR / "combined_train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "combined_test.csv", index=False)

    metadata = {
        "sources": all_data['source'].value_counts().to_dict(),
        "label_mapping": {"positive": 1, "negative": 0},
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df)
    }
    with open(PROCESSED_DATA_DIR / "combined_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Data preparation complete.")

if __name__ == "__main__":
    main()
