import requests
import json

def test_api():
    # API endpoint
    url = "http://127.0.0.1:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        health_response = requests.get(f"{url}/health")
        print(f"Health check response: {health_response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing health endpoint: {e}")
    
    # Test predict endpoint
    print("\nTesting predict endpoint...")
    test_text = "This movie was fantastic! I really enjoyed every moment of it."
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "text": test_text
    }
    
    try:
        # Make the POST request
        predict_response = requests.post(
            f"{url}/predict",
            headers=headers,
            json=data
        )
        
        # Print response details
        print(f"Response status code: {predict_response.status_code}")
        print(f"Response headers: {predict_response.headers}")
        print(f"Response body: {json.dumps(predict_response.json(), indent=2)}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error testing predict endpoint: {e}")

if __name__ == "__main__":
    test_api() 