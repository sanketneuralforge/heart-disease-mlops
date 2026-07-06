"""
tests/test_data_processing.py

Unit tests for the preprocessing pipeline saved by the EDA notebook (Task 1)
and consumed by src/train.py and app/main.py. Verifies imputation, encoding,
and scaling behave correctly and are reusable outside the notebook.
"""

import numpy as np


def test_preprocessor_loads(preprocessor):
    """The saved preprocessor artifact should load without error."""
    assert preprocessor is not None


def test_preprocessor_transforms_valid_row(preprocessor, sample_raw_row):
    """A complete, valid patient record should transform without error."""
    transformed = preprocessor.transform(sample_raw_row)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0


def test_preprocessor_handles_missing_values(preprocessor, sample_raw_row_with_missing):
    """
    Records with missing ca/thal (a real characteristic of this dataset) must
    still transform successfully, since the pipeline includes an imputer.
    """
    transformed = preprocessor.transform(sample_raw_row_with_missing)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any(), "Transformed output should contain no NaNs"


def test_preprocessor_output_has_no_nans(preprocessor, sample_raw_row):
    """Output of the full pipeline (impute + scale + encode) must be fully numeric, no NaNs."""
    transformed = preprocessor.transform(sample_raw_row)
    assert not np.isnan(transformed).any()


def test_preprocessor_deterministic(preprocessor, sample_raw_row):
    """Transforming the same input twice should give identical output (no randomness)."""
    t1 = preprocessor.transform(sample_raw_row)
    t2 = preprocessor.transform(sample_raw_row)
    np.testing.assert_array_almost_equal(
        np.asarray(t1.todense() if hasattr(t1, "todense") else t1),
        np.asarray(t2.todense() if hasattr(t2, "todense") else t2),
    )


def test_preprocessor_rejects_missing_column(preprocessor, sample_raw_row):
    """Dropping a required column should raise an error rather than silently produce garbage."""
    incomplete_row = sample_raw_row.drop(columns=["age"])
    try:
        preprocessor.transform(incomplete_row)
        raised = False
    except (KeyError, ValueError):
        raised = True
    assert raised, "Expected an error when a required feature column is missing"