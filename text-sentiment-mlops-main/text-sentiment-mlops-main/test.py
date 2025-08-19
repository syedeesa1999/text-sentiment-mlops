from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import mlflow.pyfunc
import pandas as pd
import io
import logging
import os
import webbrowser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/saleemsalik786/my-first-repo.mlflow")
mlflow.set_tracking_uri(tracking_uri)
# Model status
model_loaded = False

MODEL_URI_LOCAL = "/Users/saliksaleem/.mlflow/models/<hash>/artifacts/model"  # Update with correct path

# Check for local model
if os.path.exists(MODEL_URI_LOCAL):
    MODEL_URI = MODEL_URI_LOCAL
    logger.info(f"Using local model at {MODEL_URI}")
else:
    MODEL_URI = MODEL_URI_REMOTE
    logger.info(f"Local model not found, using remote {MODEL_URI}")

# Load model
try:
   
    model = mlflow.pyfunc.load_model()
    model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/model-status")
async def model_status():
    return {"loaded": model_loaded}

@app.post("/analyze")
async def analyze_comments(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading")
    try:
        if not file.filename.endswith(('.txt', '.csv')):
            raise HTTPException(status_code=400, detail="Only .txt or .csv supported")
        content = await file.read()
        if file.filename.endswith('.csv'):
            comments = pd.read_csv(io.StringIO(content.decode('utf-8'))).iloc[:, 0].tolist()
        else:
            comments = content.decode('utf-8').splitlines()
        comments = [c.strip() for c in comments if c.strip()]
        if not comments:
            raise HTTPException(status_code=400, detail="No valid comments")
        predictions = model.predict(comments)
        sentiments = ['Positive' if pred == 1 else 'Negative' for pred in predictions]
        total_comments = len(comments)
        positive_count = sum(1 for s in sentiments if s == 'Positive')
        negative_count = total_comments - positive_count
        positive_percentage = (positive_count / total_comments * 100) if total_comments > 0 else 0
        negative_percentage = (negative_count / total_comments * 100) if total_comments > 0 else 0
        pos_neg_ratio = positive_count / negative_count if negative_count > 0 else float('inf')
        comment_list = [{"text": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, sentiments)]
        return {
            "total_comments": total_comments,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": round(positive_percentage, 2),
            "negative_percentage": round(negative_percentage, 2),
            "pos_neg_ratio": round(pos_neg_ratio, 2) if pos_neg_ratio != float('inf') else "N/A",
            "comments": comment_list
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.on_event("startup")
async def startup_event():
    webbrowser.open("http://localhost:8000")