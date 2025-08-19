import os
import json
import argparse
import pandas as pd
import dagshub
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.pytorch
dagshub.init(repo_owner='saleemsalik786', repo_name='my-first-repo', mlflow=True)
# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "experiments"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model type to class mapping
MODEL_CLASSES = {
    "distilbert": (DistilBertForSequenceClassification, DistilBertTokenizer),
    "bert": (BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaForSequenceClassification, RobertaTokenizer)
}

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

def load_data(train_sample_size: float, test_sample_size: float):
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "imdb_train_processed.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "imdb_test_processed.csv")
    
    train_df['labels'] = train_df['sentiment'].map({'positive': 1, 'negative': 0})
    test_df['labels'] = test_df['sentiment'].map({'positive': 1, 'negative': 0})
    
    if train_sample_size < 1.0:
        train_df = train_df.sample(frac=train_sample_size, random_state=42)
        print(f"Sampled training data: {len(train_df)} rows")
    
    if test_sample_size < 1.0:
        test_df = test_df.sample(frac=test_sample_size, random_state=42)
        print(f"Sampled test data: {len(test_df)} rows")
    
    return train_df[['text', 'labels']], test_df[['text', 'labels']]

def train_model(train_sample_size: float, test_sample_size: float, hyperparams: dict, config_path: str, model_type: str, pretrained_model_name: str):
    start_time = datetime.now()
    
    train_df, test_df = load_data(train_sample_size, test_sample_size)
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
    model = model_class.from_pretrained(pretrained_model_name, num_labels=2)
    
    max_length = hyperparams.pop("max_length", 128)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_length)
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(['text'])
    test_dataset = test_dataset.remove_columns(['text'])
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    default_args = {
        "output_dir": str(OUTPUT_DIR),
        "evaluation_strategy": "epoch",
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "logging_dir": str(OUTPUT_DIR / "logs"),
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "fp16": True,
        "report_to": "none",
    }
    
    args = {**default_args, **hyperparams}
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args["output_dir"] = str(OUTPUT_DIR / exp_id)
    
    training_args = TrainingArguments(**args)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"exp_{exp_id}"):
        print(f"Starting training with {len(train_df)} samples...")
        trainer.train()
        
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        training_time = (datetime.now() - start_time).total_seconds()
        mlflow.autolog()
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("pretrained_model_name", pretrained_model_name)
        mlflow.log_param("train_sample_size", train_sample_size)
        mlflow.log_param("test_sample_size", test_sample_size)
        mlflow.log_param("max_length", max_length)
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)
        
        # Log metrics
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))
        for key, value in eval_results.items():
            mlflow.log_metric(key, value)
        
        # Log model
        model_path = args["output_dir"]
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=f"{model_type}_Sentiment_{exp_id}"
        )
        
        print(f"Experiment {exp_id} completed in {training_time:.2f} seconds.")
        print(f"Results: {eval_results}")
    
    return eval_results, exp_id

def main():
    parser = argparse.ArgumentParser(description="Train a specified model for sentiment analysis with MLflow")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_sample_size = config.get("train_sample_size", 0.1)
    test_sample_size = config.get("test_sample_size", 0.1)
    hyperparams = config.get("hyperparams", {})
    max_length = config.get("max_length", 128)
    model_type = config.get("model_type", "distilbert")
    pretrained_model_name = config.get("pretrained_model_name", "distilbert-base-uncased")
    
    hyperparams["max_length"] = max_length
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/saleemsalik786/my-first-repo.mlflow/")
    mlflow.set_experiment("DistillBert")
    
    train_model(train_sample_size, test_sample_size, hyperparams, args.config, model_type, pretrained_model_name)

if __name__ == "__main__":
    main()