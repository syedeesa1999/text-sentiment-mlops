import requests
import json

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint():
    """Test the sentiment prediction endpoint"""
    # Test with a simple positive text
    text = {"text": "This is good!"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(text)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result


def test_predict_endpoint_invalid_input():
    """Test the predict endpoint with invalid input"""
    # Test with empty text
    empty_text = {"text": ""}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(empty_text)
    )
    assert response.status_code == 400


def test_predict_endpoint_wrong_method():
    """Test that GET method is not allowed for predict endpoint"""
    response = requests.get(f"{BASE_URL}/predict")
    assert response.status_code == 405


def test_predict_endpoint_edge_cases():
    """Test the predict endpoint with various edge cases"""
    # Test with very long text
    long_text = {"text": "This is a very long text " * 100}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(long_text)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result

    # Test with special characters
    special_chars = {"text": "!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./~"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(special_chars)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result

    # Test with numbers
    numbers = {"text": "1234567890"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(numbers)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result


def test_predict_endpoint_multilingual():
    """Test the predict endpoint with different languages"""
    # Test with Spanish text
    spanish_text = {"text": "¡Este producto es excelente!"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(spanish_text)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result

    # Test with French text
    french_text = {"text": "Ce produit est terrible!"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(french_text)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result


def test_predict_endpoint_content_types():
    """Test the predict endpoint with different content types"""
    # Test with text/plain
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "text/plain"},
        data="This is a test"
    )
    assert response.status_code == 400

    # Test with application/x-www-form-urlencoded
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data="text=This is a test"
    )
    assert response.status_code == 400


def test_predict_endpoint_batch_requests():
    """Test the predict endpoint with rapid consecutive requests"""
    text = {"text": "This is a test"}
    responses = []
    for _ in range(5):
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(text)
        )
        responses.append(response)
    
    # Check all responses are successful
    for response in responses:
        assert response.status_code == 200
        result = response.json()
        assert "sentiment" in result
        assert "confidence" in result


def test_predict_endpoint_unicode():
    """Test the predict endpoint with Unicode characters"""
    unicode_text = {"text": "Hello 世界! 🌍"}
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(unicode_text)
    )
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result 