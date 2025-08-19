import os
import logging
import io
import base64
from io import BytesIO
from collections import Counter
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from wordcloud import WordCloud
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from fastapi import UploadFile, File
from transformers import DistilBertTokenizerFast
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import re
from prometheus_client import start_http_server, Counter as PrometheusCounter
import threading

REQUEST_COUNT = PrometheusCounter("request_count", "Total number of requests")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/saleemsalik786/my-first-repo.mlflow"
)
REGISTERED_MODEL_NAME = "top_perform_model"
MODEL_STAGE = "latest"
MAX_LENGTH = 256

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Sentiment Analysis Service")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- Globals ----------------
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Helper Functions ----------------
def clean_text_basic(text: str) -> str:
    import re
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('<br />', ' ').replace('<br>', ' ')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def softmax(logits: np.ndarray) -> np.ndarray:
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def load_model_and_tokenizer():
    global model, tokenizer
    logger.info("Loading model and tokenizer from MLflow registry...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.to(device).eval()
    model.config.output_attentions = True  # enable attention
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    logger.info("Model and tokenizer loaded successfully")

def predict_sentiment(text: str) -> str:
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not loaded")
    processed = clean_text_basic(text)
    enc = tokenizer(processed, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()[0]
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    return "positive" if pred == 1 else "negative"

def highlight_sentiment_words(text: str):
    processed = clean_text_basic(text)
    tokens = tokenizer.tokenize(processed)
    enc = tokenizer(processed, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        output = model(**enc)
        if not hasattr(output, 'attentions') or output.attentions is None:
            return processed

        # Average over last 4 layers and all heads
        attn_layers = output.attentions[-4:]  # list of tensors [1, heads, seq_len, seq_len]
        mean_attn = torch.stack(attn_layers).mean(0).squeeze(0).mean(0).mean(0)  # mean over layers, heads, tokens
        attn_weights = mean_attn[:len(tokens)].cpu().numpy()
        attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min() + 1e-8)

    html = ""
    for token, weight in zip(tokens, attn_weights):
        word = token.replace("##", "")
        if word in stop_words:
            weight = 0.0
        color = int(255 * (1 - weight))
        html += f'<span style="background-color:rgb(255,{color},{color}); padding:2px; border-radius:4px; margin:1px;">{word}</span> '
    return html

def generate_wordcloud_and_top_words(comments):
    # Clean and split text
    words = []
    for text in comments:
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words.extend([word for word in cleaned.split() if word not in stop_words])

    # Count word frequencies
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)  # top 10 frequent words

    # Generate word cloud image
    wc = WordCloud(width=400, height=300, background_color="white").generate_from_frequencies(word_counts)
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    encoded_wc = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded_wc, top_words

# ---------------- Startup ----------------
@app.on_event("startup")
def startup_event():
    try:
        threading.Thread(target=lambda: start_http_server(8001)).start()
        load_model_and_tokenizer()
    except Exception as e:
        logger.error(f"Startup error: {e}")

# ---------------- Health ----------------
@app.get("/health")
def health():
    status = "ok" if model and tokenizer else "loading"
    logger.info(f"Health check status: {status}")
    return {"status": status}

# ---------------- Home ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logger.info("Rendering home page")
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# ---------------- Single Predict ----------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, review: str = Form(...)):
    logger.info("Received single review prediction request")
    if not model or not tokenizer:
        logger.error("Model or tokenizer not loaded for single predict")
        raise HTTPException(503, "Model not loaded")
    text = review.strip()
    if not text:
        logger.warning("Empty review provided")
        raise HTTPException(400, "Empty review")
    processed = clean_text_basic(text)
    enc = tokenizer(processed, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.cpu().numpy()[0]
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[pred]
    logger.info(f"Single prediction result: {sentiment} ({confidence:.2%})")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": True, "review": text, "sentiment": sentiment, "confidence": f"{confidence:.2%}"}
    )

# ---------------- Bulk Predict ----------------
@app.get("/bulk", response_class=HTMLResponse)
def bulk_form(request: Request):
    logger.info("Rendering bulk analysis form")
    
    return templates.TemplateResponse("bulk.html", {"request": request, "processed": None})

@app.post("/bulk", response_class=HTMLResponse)
async def bulk_predict(request: Request, comments: str = Form(...), file: UploadFile = File(None)):
    REQUEST_COUNT.inc()
    logger.info("Received request for bulk sentiment analysis.")
    text_data = ""

    # 1. Try file input first
    if file is not None:
        contents = await file.read()
        try:
            text_data = contents.decode("utf-8").strip()
        except Exception as e:
            logger.error(f"Error decoding file: {e}")
            raise HTTPException(400, "Invalid file encoding")

    # 2. If no file or file empty, try textarea
    if not text_data and comments is not None:
        text_data = comments.strip()

    if not text_data:
        raise HTTPException(400, "No input text or file provided")
    lines = [line.strip() for line in text_data.splitlines() if line.strip()]
    logger.info(f"Number of comments to analyze: {len(lines)}")
    
    results = []
    pos = 0
    neg = 0
    for idx, comment in enumerate(lines):
        logger.info(f"Processing comment #{idx+1}: {comment[:50]}...")
        try:
            sentiment = predict_sentiment(comment)
            # highlight = highlight_sentiment_words(comment)
            
        except Exception as e:
            logger.error(f"Error processing comment #{idx+1}: {e}")
            sentiment = "error"
        logger.info(f" → Sentiment: {sentiment}")
        if sentiment == "positive":
            pos += 1
        elif sentiment == "negative":
            neg += 1
        results.append({"text": comment, "sentiment": sentiment})
        
    # Pie Chart Generation
    wordcloud_img, top_words = generate_wordcloud_and_top_words(lines)
    logger.info("Generating pie chart...")
    fig, ax = plt.subplots()
    if pos + neg > 0:
        ax.pie([pos, neg], labels=[f"Positive ({pos})", f"Negative ({neg})"], autopct="%1.1f%%")
    else:
        ax.text(0.5, 0.5, 'No sentiments detected', ha='center', va='center', fontsize=12)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    plt.close()
    chart_base64 = base64.b64encode(img_bytes).decode()

    logger.info("Sentiment analysis completed and response ready.")
    return templates.TemplateResponse("bulk.html", {
        "request": request,
        "processed": True,
        "comments": comments,
        "results": results,
        "chart": chart_base64,
        "wordcloud_img": wordcloud_img,
        "top_words": top_words
    })

# ---------------- Run ----------------
if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)