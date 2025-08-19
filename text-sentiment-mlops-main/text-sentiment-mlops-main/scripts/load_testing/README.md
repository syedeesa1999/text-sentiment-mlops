# Load Testing Setup

This directory contains load testing scripts for the sentiment analysis API using Locust.

## Prerequisites

- Python 3.8+
- Locust (install with `pip install -r requirements.txt`)

## Usage

1. Start the API server first:
```bash
python scripts/model_serving/api.py
```

2. In a new terminal, start the Locust load testing server:
```bash
cd scripts/load_testing
locust -f locustfile.py
```

3. Open your browser and go to http://localhost:8089

4. Configure the load test:
   - Number of users: Number of concurrent users to simulate
   - Spawn rate: How many users to spawn per second
   - Host: http://localhost:8000 (or your API URL)

5. Click "Start swarming" to begin the load test

## Test Scenarios

The load test includes:

1. Sentiment Prediction (Weight: 3)
   - POST requests to `/predict` endpoint
   - Random text samples from predefined list
   - Tests model performance under load

2. Health Check (Weight: 1)
   - GET requests to `/health` endpoint
   - Monitors API availability

3. Statistics (Weight: 1)
   - GET requests to `/stats` endpoint
   - Tests monitoring system performance

4. Dashboard (Weight: 1)
   - GET requests to `/dashboard` endpoint
   - Tests static file serving

## Metrics

Locust provides real-time metrics including:
- Request count
- Response times
- Failure rates
- Number of users
- RPS (Requests Per Second)

## Customization

You can modify the load test by:
1. Adjusting `wait_time` in `locustfile.py`
2. Changing task weights
3. Adding more sample texts
4. Modifying the number of users and spawn rate

## Best Practices

1. Start with a small number of users and gradually increase
2. Monitor system resources during testing
3. Keep test duration reasonable (5-10 minutes)
4. Document any failures or bottlenecks
5. Test different scenarios (normal load, peak load, stress test) 