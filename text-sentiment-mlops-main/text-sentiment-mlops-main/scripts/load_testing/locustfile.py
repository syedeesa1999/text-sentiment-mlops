from locust import HttpUser, task, between
import random
import json
from collections import defaultdict

# Sample texts for testing
SAMPLE_TEXTS = [
    "This is a great product! I love it!",
    "Terrible service, would not recommend.",
    "The movie was okay, but nothing special.",
    "Amazing experience, exceeded my expectations!",
    "Very disappointed with the quality.",
    "Best purchase I've made this year!",
    "Not worth the money at all.",
    "Excellent customer service and fast delivery.",
    "The product broke after just one use.",
    "I'm very satisfied with my purchase."
]

class SentimentAnalysisUser(HttpUser):
    # Wait between 1 to 3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        # Check if the API is healthy
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                instance_id = data.get("instance_info", {}).get("id", "unknown")
                print(f"Connected to instance {instance_id}")
                response.success()
            else:
                response.failure(f"Health check failed with status code: {response.status_code}")
    
    @task(3)  # Higher weight for prediction endpoint
    def predict_sentiment(self):
        """Test the sentiment prediction endpoint"""
        text = random.choice(SAMPLE_TEXTS)
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                instance_id = data.get("instance_info", {}).get("id", "unknown")
                print(f"Request handled by instance {instance_id}")
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")
    
    @task(1)  # Lower weight for health check
    def health_check(self):
        """Test the health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                instance_id = data.get("instance_info", {}).get("id", "unknown")
                print(f"Health check from instance {instance_id}")
                response.success()
            else:
                response.failure(f"Health check failed with status code: {response.status_code}")
    
    @task(1)  # Lower weight for stats endpoint
    def get_stats(self):
        """Test the statistics endpoint"""
        with self.client.get("/stats", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                instance_id = data.get("instance_info", {}).get("id", "unknown")
                print(f"Stats from instance {instance_id}")
                response.success()
            else:
                response.failure(f"Stats endpoint failed with status code: {response.status_code}")
    
    @task(1)  # Lower weight for dashboard
    def get_dashboard(self):
        """Test the dashboard endpoint"""
        with self.client.get("/dashboard", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboard failed with status code: {response.status_code}") 