"""
tests/test_model.py

Unit tests for the final packaged model pipeline (models/final_model.joblib),
produced by src/train.py. Verifies it loads standalone and produces valid,
well-formed predictions directly on raw (untransformed) patient data —
exactly how app/main.py will use it.
"""

import numpy as np


def test_model_loads(final_model):
    """The saved final model pipeline should load without error."""
    assert final_model is not None


def test_model_predicts_valid_class(final_model, sample_raw_row):
    """Predictions must be one of the two valid classes: 0 (no disease) or 1 (disease)."""
    pred = final_model.predict(sample_raw_row)
    assert pred[0] in (0, 1)


def test_model_predict_proba_valid_range(final_model, sample_raw_row):
    """Predicted probabilities must be valid: two classes, each in [0, 1], summing to 1."""
    proba = final_model.predict_proba(sample_raw_row)
    assert proba.shape == (1, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_almost_equal(proba.sum(), 1.0, decimal=5)


def test_model_predict_matches_argmax_proba(final_model, sample_raw_row):
    """The predicted class should correspond to the higher of the two probabilities."""
    pred = final_model.predict(sample_raw_row)[0]
    proba = final_model.predict_proba(sample_raw_row)[0]
    assert pred == np.argmax(proba)


def test_model_handles_missing_optional_fields(final_model, sample_raw_row_with_missing):
    """
    End-to-end check that the full pipeline (imputer + scaler/encoder + model)
    handles a record with missing ca/thal without raising an error.
    """
    pred = final_model.predict(sample_raw_row_with_missing)
    proba = final_model.predict_proba(sample_raw_row_with_missing)
    assert pred[0] in (0, 1)
    assert proba.shape == (1, 2)


def test_model_deterministic_predictions(final_model, sample_raw_row):
    """The same input should always produce the same prediction (no randomness at inference)."""
    pred1 = final_model.predict(sample_raw_row)
    pred2 = final_model.predict(sample_raw_row)
    assert pred1[0] == pred2[0]