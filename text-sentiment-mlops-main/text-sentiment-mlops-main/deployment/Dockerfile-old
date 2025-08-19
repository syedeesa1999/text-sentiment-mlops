# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/saved/sentiment_model \
    TOKENIZER_PATH=/app/models/saved/sentiment_tokenizer \
    LOG_DIR=/app/logs

# Create necessary directories
RUN mkdir -p /app/models/saved/sentiment_model \
    /app/models/saved/sentiment_tokenizer \
    /app/logs \
    /app/static

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire scripts directory
COPY scripts /app/scripts/

# Create symbolic link for monitoring module
RUN ln -s /app/scripts/monitoring /app/monitoring

# Copy the API files
COPY scripts/model_serving/api.py /app/api.py
COPY scripts/model_serving/predict.py /app/predict.py

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]