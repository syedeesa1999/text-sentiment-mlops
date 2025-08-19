# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY scripts/API/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, templates, and static files
COPY scripts/API/ ./

# Expose ports for FastAPI and Prometheus metrics
EXPOSE 8000 8001

# Default environment variables
ENV MLFLOW_TRACKING_URI=https://dagshub.com/saleemsalik786/my-first-repo.mlflow \
    REGISTERED_MODEL_NAME=top_perform_model \
    MODEL_STAGE=latest

# Default command to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]