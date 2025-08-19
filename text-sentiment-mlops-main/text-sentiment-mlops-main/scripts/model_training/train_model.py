import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import json

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    print("Starting model training...")
    os.makedirs("models/saved", exist_ok=True)

    # Load combined multi-dataset data
    train_df = pd.read_csv("data/processed/combined_train.csv")
    test_df = pd.read_csv("data/processed/combined_test.csv")

    # Convert sentiment to numeric labels
    label_map = {'positive': 1, 'negative': 0}
    train_df['labels'] = train_df['sentiment'].map(label_map).astype('int64')
    test_df['labels'] = test_df['sentiment'].map(label_map).astype('int64')

    train_df = train_df[['text', 'labels']]
    test_df = test_df[['text', 'labels']]

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        problem_type="single_label_classification"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    train_dataset.set_format('torch')
    test_dataset.set_format('torch')

    training_args = TrainingArguments(
        output_dir='models/saved/results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='models/saved/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    print("Evaluating model...")
    eval_result = trainer.evaluate()

    print("Saving model and tokenizer...")
    model.save_pretrained("models/saved/sentiment_model")
    tokenizer.save_pretrained("models/saved/sentiment_tokenizer")

    with open("models/saved/eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)

    print("\n✅ Training complete. Model and tokenizer saved.")
    print(f"Evaluation results: {eval_result}")

if __name__ == "__main__":
    train_model()
