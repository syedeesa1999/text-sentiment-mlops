import pytest
import requests
import time


@pytest.fixture(scope="session", autouse=True)
def ensure_api_running():
    """Ensure the API is running before running tests"""
    base_url = "http://localhost:8000"
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception("API is not running. Please start the API server before running tests.") 