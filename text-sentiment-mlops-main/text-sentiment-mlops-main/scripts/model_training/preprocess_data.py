import pandas as pd
import numpy as np
import os

def preprocess_data():
    print("Starting data preprocessing...")
    
    # Create processed data directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Load raw data
    train_df = pd.read_csv("data/raw/imdb_train.csv")
    test_df = pd.read_csv("data/raw/imdb_test.csv")
    
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    print("\nColumns in the dataset:")
    print(train_df.columns.tolist())
    
    # Clean and preprocess the data
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Convert to string
        text = str(text)
        # Remove HTML tags
        text = text.replace('<br />', ' ')
        # Remove special characters and extra whitespace
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())
        return text
    
    # Clean text data
    print("\nCleaning text data...")
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    
    # Remove empty texts
    train_df = train_df[train_df['text'].str.len() > 0]
    test_df = test_df[test_df['text'].str.len() > 0]
    
    # Convert label to sentiment
    print("Processing sentiment labels...")
    train_df['sentiment'] = train_df['label'].map({1: 'positive', 0: 'negative'})
    test_df['sentiment'] = test_df['label'].map({1: 'positive', 0: 'negative'})
    
    # Remove rows with invalid sentiment labels
    train_df = train_df.dropna(subset=['sentiment'])
    test_df = test_df.dropna(subset=['sentiment'])
    
    # Keep only necessary columns
    train_df = train_df[['text', 'sentiment']]
    test_df = test_df[['text', 'sentiment']]
    
    # Save processed data
    print("Saving processed data...")
    train_df.to_csv("data/processed/imdb_train_processed.csv", index=False)
    test_df.to_csv("data/processed/imdb_test_processed.csv", index=False)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print("\nSentiment distribution in training set:")
    print(train_df['sentiment'].value_counts())
    print("\nSentiment distribution in test set:")
    print(test_df['sentiment'].value_counts())
    
    print("\nPreprocessing complete. Processed data saved to data/processed/")

if __name__ == "__main__":
    preprocess_data() 