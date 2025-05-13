from fastapi.testclient import TestClient
import sys
import os
os.environ["API_TOKEN"] = "test-secret"
# Add app path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
# Set up environment variable for testing

client = TestClient(app)

# === Tests for /hello endpoint ===

def test_hello_with_valid_name():
    response = client.post(
        "/hello", 
        json={"name": "John"}, 
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Hello John"}

def test_hello_with_empty_name():
    response = client.post(
        "/hello", 
        json={"name": ""}, 
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Hello "}

def test_hello_missing_name_field():
    response = client.post(
        "/hello", 
        json={}, 
    )
    assert response.status_code == 422

def test_hello_with_non_string_name():
    response = client.post(
        "/hello", 
        json={"name": 123}, 
    )
    assert response.status_code == 422

# === Tests for /predict endpoint ===

def test_predict_valid_input_case1():
    response = client.post(
        "/predict", 
        json={
            "petal_length": 2,
            "sepal_length": 2,
            "petal_width": 0.5,
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 200
    assert "predicted_species" in response.json()

def test_predict_valid_input_case2():
    response = client.post(
        "/predict", 
        json={
            "petal_length": 1,
            "sepal_length": 4,
            "petal_width": 2,
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 200
    assert "predicted_species" in response.json()

def test_predict_missing_feature():
    response = client.post(
        "/predict", 
        json={
            "petal_length": 1,
            "sepal_length": 4,
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 422  # Missing petal_width

def test_predict_string_instead_of_float():
    response = client.post(
        "/predict", 
        json={
            "petal_length": 1,
            "sepal_length": "4",
            "petal_width": "2",
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 200  # Still valid, due to automatic coercion

def test_predict_invalid_string_input():
    response = client.post(
        "/predict", 
        json={
            "petal_length": "Hallo",
            "sepal_length": "BIPM",
            "petal_width": 2,
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "test-secret"}
    )
    assert response.status_code == 422  # Invalid strings for float fields

def test_predict_invalid_token():
    response = client.post(
        "/predict", 
        json={
            "petal_length": 1,
            "sepal_length": 4,
            "petal_width": 2,
            "sepal_width": 3
        }, 
        headers={"X-API-Token": "wrong-token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
