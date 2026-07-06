"""
tests/test_api.py

Unit tests for the FastAPI serving layer (app/main.py). Verifies the
/health and /predict endpoints behave correctly for valid input, input
with missing optional fields, and malformed/invalid requests.
"""


def test_health_endpoint_returns_healthy(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_root_endpoint(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_valid_payload_returns_200(api_client, valid_api_payload):
    response = api_client.post("/predict", json=valid_api_payload)
    assert response.status_code == 200


def test_predict_response_schema(api_client, valid_api_payload):
    """Response must contain all expected fields with correct types/ranges."""
    response = api_client.post("/predict", json=valid_api_payload)
    body = response.json()

    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("No Disease", "Disease Present")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_with_missing_optional_fields(api_client, valid_api_payload):
    """ca/thal are Optional; a null value should still produce a valid prediction."""
    payload = dict(valid_api_payload)
    payload["ca"] = None
    payload["thal"] = None

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)


def test_predict_missing_required_field_returns_422(api_client, valid_api_payload):
    """Dropping a required field (e.g. age) should fail validation, not crash the server."""
    payload = dict(valid_api_payload)
    del payload["age"]

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(api_client, valid_api_payload):
    """Sending a string where a number is expected should fail validation cleanly."""
    payload = dict(valid_api_payload)
    payload["age"] = "not_a_number"

    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_body_returns_422(api_client):
    response = api_client.post("/predict", json={})
    assert response.status_code == 422