import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pathlib import Path
import logging
import csv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/Users/saliksaleem/Desktop/Projects/text-sentiment-mlops/models/experiments/serve.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
EXPERIMENTS_DIR = Path("/Users/saliksaleem/Desktop/Projects/text-sentiment-mlops/models/experiments")
LOG_FILE = EXPERIMENTS_DIR / "experiment_log.csv"
PREDICTION_LOG = EXPERIMENTS_DIR / "predictions.csv"
HTML_DIR = Path("/Users/saliksaleem/Desktop/Projects/text-sentiment-mlops")

# Pydantic model for request body
class PredictRequest(BaseModel):
    text: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for demo; restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files (e.g., predict.html)
app.mount("/static", StaticFiles(directory=HTML_DIR), name="static")

def get_best_model(metric="eval_f1"):
    """Load the model with the highest F1 score from experiment_log.csv."""
    if not LOG_FILE.exists():
        logger.error(f"Experiment log not found at {LOG_FILE}")
        raise FileNotFoundError(f"Experiment log not found at {LOG_FILE}")
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        logger.error("No experiments found in log")
        raise ValueError("No experiments found in log")
    best_exp_id = df.loc[df[metric].idxmax()]["experiment_id"]
    model_path = EXPERIMENTS_DIR / best_exp_id
    if not model_path.exists():
        logger.error(f"Model directory not found at {model_path}")
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    return model_path, best_exp_id

# Load best model
try:
    model_path, current_model_id = get_best_model()
    logger.info(f"Loading model from {model_path} (experiment_id: {current_model_id})")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a welcome message or redirect to predict.html."""
    return """
    <html>
        <head><title>Sentiment Analysis Demo</title></head>
        <body>
            <h2>Welcome to the Sentiment Analysis Demo</h2>
            <p>Visit <a href="/static/predict.html">Predict Sentiment</a> to try the model.</p>
            <p>View <a href="/experiments">Experiment Results</a> to see all experiments.</p>
            <p>Use POST /predict for API predictions.</p>
        </body>
    </html>
    """

@app.get("/predict")
async def predict_get():
    """Provide instructions for using the predict endpoint."""
    return {
        "message": "Use POST /predict with a JSON body {'text': 'your review'} to get sentiment predictions.",
        "example": "curl -X POST http://localhost:8000/predict -d '{\"text\": \"This movie was amazing!\"}'"
    }

@app.get("/experiments", response_class=HTMLResponse)
async def get_experiments():
    """Return an HTML page with a table of experiments, marking the best and current model."""
    try:
        if not LOG_FILE.exists():
            logger.error(f"Experiment log not found at {LOG_FILE}")
            return "<html><body><h2>Error</h2><p>Experiment log not found.</p></body></html>"
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            logger.error("No experiments found in log")
            return "<html><body><h2>Error</h2><p>No experiments found in log.</p></body></html>"
        
        # Identify the best experiment
        best_exp_id = df.loc[df["eval_f1"].idxmax()]["experiment_id"]
        
        # Build HTML table
        table_rows = ""
        for _, row in df.iterrows():
            is_best = "Yes" if row["experiment_id"] == best_exp_id else "No"
            table_rows += (
                f"<tr>"
                f"<td>{row['experiment_id']}</td>"
                f"<td>{row['eval_accuracy']:.3f}</td>"
                f"<td>{row['eval_f1']:.3f}</td>"
                f"<td>{row['eval_precision']:.3f}</td>"
                f"<td>{row['eval_recall']:.3f}</td>"
                f"<td>{row['training_time_seconds']:.2f}</td>"
                f"<td>{int(row['train_samples'])}</td>"
                f"<td>{is_best}</td>"
                f"</tr>"
            )
        
        html_content = f"""
        <html>
            <head>
                <title>Experiment Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; max-width: 1000px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .best {{ font-weight: bold; color: #007bff; }}
                    p {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h2>Experiment Results</h2>
                <p>Current Model: Experiment {current_model_id}</p>
                <p>Best Experiment: {best_exp_id} (based on F1 score)</p>
                <table>
                    <tr>
                        <th>Experiment ID</th>
                        <th>Accuracy</th>
                        <th>F1 Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Training Time (s)</th>
                        <th>Train Samples</th>
                        <th>Best</th>
                    </tr>
                    {table_rows}
                </table>
                <p><a href="/">Back to Home</a> | <a href="/static/predict.html">Predict Sentiment</a></p>
            </body>
        </html>
        """
        logger.info("Rendered experiment results HTML")
        return html_content
    except Exception as e:
        logger.error(f"Error rendering experiments: {str(e)}")
        return f"<html><body><h2>Error</h2><p>{str(e)}</p></body></html>"

@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict sentiment using the best model and log the result."""
    try:
        text = request.text
        if not text.strip():
            logger.warning("Empty input text received")
            return {"error": "Input text cannot be empty"}
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
        
        # Log prediction
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "prediction": "positive" if prediction == 1 else "negative",
            "confidence": confidence,
            "model_id": current_model_id
        }
        file_exists = PREDICTION_LOG.exists()
        with open(PREDICTION_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
        
        logger.info(f"Prediction made: {log_entry['prediction']} (confidence: {confidence:.3f})")
        return {
            "sentiment": log_entry["prediction"],
            "confidence": confidence,
            "model_id": current_model_id
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)