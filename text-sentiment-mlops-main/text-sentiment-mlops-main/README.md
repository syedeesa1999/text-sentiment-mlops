# Text Sentiment Analysis with MLOps

A machine learning project that performs sentiment analysis on text using DistilBERT, with a modern web interface and MLOps practices.

## Project Overview

This project implements a sentiment analysis system that:
- Uses DistilBERT for text classification
- Provides a modern web interface for user interaction
- Implements MLOps best practices
- Includes data processing, model training, and serving pipelines
- Features real-time monitoring and performance tracking
- Supports containerized deployment with Docker
- Includes load balancing across multiple instances
- Features comprehensive load testing capabilities
- Implements CI/CD pipeline with GitHub Actions

## Project Structure

```
text-sentiment-mlops/
├── data/
│   └── processed/
│       ├── imdb_train_processed.csv
│       ├── imdb_test_processed.csv
│       └── metadata.json
├── models/
│   └── saved/
│       ├── sentiment_model/
│       │   ├── model.safetensors
│       │   └── config.json
│       └── sentiment_tokenizer/
│           ├── vocab.txt
│           ├── special_tokens_map.json
│           └── tokenizer_config.json
├── scripts/
│   ├── data_processing/
│   │   └── prepare_data.py
│   ├── model_training/
│   │   ├── preprocess_data.py
│   │   └── train_model.py
│   ├── model_serving/
│   │   ├── api.py              # FastAPI application
│   │   ├── predict.py          # Prediction logic
│   │   ├── test_api.py         # API testing
│   │   └── static/             # Static files
│   │       ├── index.html      # Main interface
│   │       └── dashboard.html  # Monitoring dashboard
│   └── monitoring/
│       └── logger.py           # Monitoring and logging
├── deployment/                  # Docker configuration
│   ├── Dockerfile              # Container definition
│   ├── docker-compose.yml      # Container orchestration
│   └── .dockerignore          # Files to exclude from build
├── .github/
│   └── workflows/             # CI/CD pipelines
│       ├── ci.yml             # Continuous Integration
│       └── cd.yml             # Continuous Deployment
├── logs/                       # Application logs
├── requirements.txt
└── README.md
```

## Recent Updates

### CI/CD Pipeline
- Added GitHub Actions workflows for CI/CD
- Implemented automated testing and linting
- Added Docker image building and pushing
- Configured deployment environment variables

### Load Balancing
- Implemented Nginx load balancing across three API instances
- Added health checks for each instance
- Configured least connections algorithm for optimal distribution
- Added instance tracking and monitoring
- Implemented automatic failover

### Load Testing
- Added Locust for load testing
- Implemented performance monitoring
- Added concurrent user simulation
- Created load testing scenarios
- Added instance-specific request tracking
- Implemented response time analysis

### Monitoring System
- Added real-time monitoring of API performance
- Implemented logging system for request tracking
- Created dashboard for visualizing metrics
- Added drift detection capabilities

### Static Files Organization
- Moved static files to dedicated directory
- Improved file structure for better maintainability
- Added modern, responsive dashboard interface
- Enhanced error handling and user feedback

### Docker Configuration
- Updated Dockerfile for better performance
- Improved volume mounting for logs and static files
- Added health checks and monitoring
- Enhanced container security
- Added multi-instance support with Nginx load balancing
- Configured container networking for load balancing

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Pandas
- scikit-learn
- uvicorn
- Docker (for containerized deployment)
- Locust (for load testing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/salik786/text-sentiment-mlops.git
cd text-sentiment-mlops
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing

Process the raw data:
```bash
python scripts/data_processing/prepare_data.py
```

### 2. Model Training

Train the sentiment analysis model:
```bash
python scripts/model_training/preprocess_data.py
python scripts/model_training/train_model.py
```

### 3. Running the Application

#### Local Development

Start the FastAPI server:
```bash
python scripts/model_serving/api.py
```
The API will be available at `http://127.0.0.1:8000`

#### Docker Deployment with Load Balancing

1. Build and run the containers:
```bash
cd deployment
docker-compose up --build
```

This will start:
- 3 API instances (sentiment-api-1, sentiment-api-2, sentiment-api-3)
- 1 Nginx load balancer
- All services will be accessible through port 8000

2. Access the application:
- API endpoints: `http://localhost:8000/predict` and `http://localhost:8000/health`
- Test deployment: `http://localhost:8000/test-deployment`
- Dashboard: `http://localhost:8000/dashboard`

3. Verify load balancing:
```bash
# Check health of all instances
curl http://localhost:8000/health

# Make multiple requests to see load distribution
for i in {1..10}; do curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"This is a test"}'; done
```

### 4. Load Testing

Run load tests using Locust:
```bash
cd scripts/load_testing
locust -f locustfile.py
```
Access the Locust web interface at `http://localhost:8089`

Configure the test:
- Number of users: 10-20
- Spawn rate: 1-2 users/second
- Host: http://localhost:8000

The test will:
- Track requests across all instances
- Monitor response times
- Show load distribution
- Display error rates
- Provide real-time metrics

### 5. Running Tests

Run the test suite:
```bash
pytest scripts/tests/
```

## Load Balancing Configuration

The system uses Nginx for load balancing with the following features:
- Least connections algorithm for optimal distribution
- Health checks for each instance
- Automatic failover
- Request tracking per instance
- Response time monitoring

### Nginx Configuration
```nginx
upstream sentiment_api {
    least_conn;  # Use least connections algorithm
    
    server sentiment-api-1:8000 max_fails=3 fail_timeout=30s;
    server sentiment-api-2:8000 max_fails=3 fail_timeout=30s;
    server sentiment-api-3:8000 max_fails=3 fail_timeout=30s;
}
```

### Instance Health Checks
Each instance provides:
- Health status
- Instance ID
- Hostname
- IP address
- Response times

## API Endpoints

- `GET /`: Main interface
- `GET /dashboard`: Monitoring dashboard
- `GET /health`: Health check endpoint
- `GET /stats`: Performance statistics
- `POST /predict`: Sentiment analysis endpoint
  ```json
  {
    "text": "Your text here"
  }
  ```

## Monitoring Features

The monitoring system provides:
- Real-time request tracking
- Performance metrics visualization
- Sentiment distribution analysis
- Response time monitoring
- Error rate tracking
- Data drift detection

## CI/CD Pipeline

The project includes:
- Automated testing with pytest
- Code linting with flake8
- Docker image building and pushing
- GitHub Actions workflows
- Environment variable management
- Deployment versioning

## Load Testing Features

The load testing system provides:
- Concurrent user simulation
- Response time analysis
- Request rate monitoring
- Error rate tracking
- Custom test scenarios
- Real-time metrics visualization

## Web Interface Features

The web interface provides:
- Modern, responsive design
- Real-time sentiment analysis
- Visual confidence indicators
- Detailed probability scores
- Loading animations
- Error handling

## Model Details

- Architecture: DistilBERT
- Task: Binary sentiment classification (positive/negative)
- Input: Text
- Output: Sentiment label, confidence score, and probability distribution

## Development

### Adding New Features

1. Data Processing:
   - Modify `scripts/data_processing/prepare_data.py`
   - Add new preprocessing steps as needed

2. Model Training:
   - Modify `scripts/model_training/train_model.py`
   - Adjust hyperparameters or model architecture

3. API:
   - Modify `scripts/model_serving/api.py`
   - Add new endpoints or modify existing ones

4. Web Interface:
   - Modify `scripts/model_serving/index.html`
   - Update the UI/UX as needed

### Testing

Run the test script to verify the API functionality:
```bash
python scripts/model_serving/test_api.py
```

## Troubleshooting

1. Port Already in Use:
   ```bash
   # Find process using port 8080
   lsof -i :8080
   # Kill the process
   kill <PID>
   ```

2. Model Loading Issues:
   - Ensure model files exist in `models/saved/`
   - Check model and tokenizer paths in `predict.py`

3. API Connection Issues:
   - Verify both servers are running
   - Check CORS settings in `api.py`
   - Ensure correct ports are being used

4. Docker Issues:
   - Ensure Docker daemon is running
   - Check container logs: `docker-compose logs`
   - Verify model files are properly mounted
   - Check environment variables in docker-compose.yml

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Common Issues and Solutions

### Local Development Issues

1. **Import Errors**
   ```bash
   ImportError: attempted relative import with no known parent package
   ```
   **Solution**: The API uses absolute imports. Make sure you're running the API from the correct directory:
   ```bash
   cd scripts/model_serving
   python api.py
   ```

2. **Hostname Resolution Error**
   ```bash
   socket.gaierror: [Errno 8] nodename nor servname provided, or not known
   ```
   **Solution**: The API will automatically fall back to localhost (127.0.0.1) if hostname resolution fails. This is normal and won't affect functionality.

3. **Model Loading Issues**
   ```bash
   Error loading model: FileNotFoundError: [Errno 2] No such file or directory
   ```
   **Solution**: Ensure the model files are in the correct location:
   ```
   models/
   └── saved/
       ├── sentiment_model/
       └── sentiment_tokenizer/
   ```

4. **Port Already in Use**
   ```bash
   OSError: [Errno 48] Address already in use
   ```
   **Solution**: Either:
   - Stop any other service using port 8000
   - Or change the port in api.py:
     ```python
     uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
     ```

5. **Missing Dependencies**
   ```bash
   ModuleNotFoundError: No module named 'transformers'
   ```
   **Solution**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. **Static Files Not Found**
   ```bash
   FileNotFoundError: [Errno 2] No such file or directory: 'static/index.html'
   ```
   **Solution**: Ensure the static directory exists with required files:
   ```
   scripts/model_serving/
   └── static/
       ├── index.html
       └── dashboard.html
   ```

7. **CUDA/GPU Issues**
   ```bash
   RuntimeError: CUDA error: out of memory
   ```
   **Solution**: The model will automatically use CPU if CUDA is not available or if there's insufficient GPU memory.

8. **Environment Variables**
   If you need to modify the default behavior, you can set these environment variables:
   ```bash
   export MODEL_PATH=/path/to/model
   export TOKENIZER_PATH=/path/to/tokenizer
   export TEST_MODE=true
   export SKIP_MODEL_LOAD=true
   export INSTANCE_ID=local-1
   ```

### Testing the API

After starting the API, you can test it using:

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Sentiment Prediction**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"I love this product!"}'
   ```

3. **Deployment Test**
   ```bash
   curl http://localhost:8000/test-deployment
   ```

4. **API Documentation**
   Open in browser: `http://localhost:8000/docs`

## Getting Started

### For New Users

1. Clone the repository:
   ```bash
   git clone https://github.com/salik786/text-sentiment-mlops.git
   cd text-sentiment-mlops
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure model files are in place:
   ```
   models/
   └── saved/
       ├── sentiment_model/
       └── sentiment_tokenizer/
   ```

4. Run the API:
   ```bash
   cd scripts/model_serving
   python api.py
   ```

### For Existing Users

1. Get the latest changes:
   ```bash
   git pull origin main
   ```

2. Install any new dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   cd scripts/model_serving
   python api.py
   ```


