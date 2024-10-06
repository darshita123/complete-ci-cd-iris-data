# test_app_live.py
import httpx
import pytest

# Replace with your deployed app's IP address
BASE_URL = "http://4.213.200.215"

# Test the root endpoint
def test_read_root():
    response = httpx.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI with a trained model!"}

# Test the predict endpoint with valid inputs
@pytest.mark.parametrize(
    "sepal_length, sepal_width, petal_length, petal_width, expected_prediction",
    [
        (5.1, 3.5, 1.4, 0.2, 0),  # Example inputs
        (6.7, 3.0, 5.2, 2.3, 1),  # Example inputs
        (5.9, 3.0, 5.1, 1.8, 2),  # Example inputs
    ]
)
def test_predict(sepal_length, sepal_width, petal_length, petal_width, expected_prediction):
    response = httpx.get(
        f"{BASE_URL}/predict",
        params={
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": expected_prediction}
