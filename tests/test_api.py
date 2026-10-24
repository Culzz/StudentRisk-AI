"""Tests for the StudentRisk AI Flask API.

Includes parity checks: the web encoder must match ``src.features.encode`` and
the API's probabilities must match XGBoost's sklearn wrapper.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, encode_values, score  # noqa: E402
from src.features import FEATURE_ORDER, FEATURE_LABELS, NUMERIC_FEATURES, BINARY_FEATURES  # noqa: E402

client = app.test_client()

SAMPLE_ROWS = [
    # age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime,
    # goout, Dalc, Walc, health, absences, G1, G2 + binaries + sex/address
    dict(age=16, Medu=4, Fedu=4, traveltime=1, studytime=3, failures=0, famrel=5,
         freetime=3, goout=2, Dalc=1, Walc=1, health=5, absences=1, G1=16, G2=17,
         schoolsup="no", famsup="yes", paid="yes", activities="yes", higher="yes",
         internet="yes", romantic="no", sex="F", address="U"),
    dict(age=18, Medu=1, Fedu=1, traveltime=3, studytime=1, failures=2, famrel=2,
         freetime=5, goout=5, Dalc=4, Walc=5, health=2, absences=40, G1=5, G2=6,
         schoolsup="yes", famsup="no", paid="no", activities="no", higher="no",
         internet="no", romantic="yes", sex="M", address="R"),
    dict(age=17, Medu=2, Fedu=2, traveltime=1, studytime=2, failures=0, famrel=4,
         freetime=3, goout=3, Dalc=1, Walc=2, health=4, absences=8, G1=10, G2=11,
         schoolsup="no", famsup="yes", paid="no", activities="yes", higher="yes",
         internet="yes", romantic="no", sex="M", address="U"),
]


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_index_and_assets_served():
    assert client.get("/").status_code == 200
    assert client.get("/assets/style.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200


def test_schema_shape():
    payload = client.get("/api/schema").get_json()
    assert len(payload["numeric"]) == 15
    assert len(payload["binaries"]) == 7
    assert len(payload["feature_order"]) == 24
    assert set(payload["presets"]) == {"high", "moderate", "low"}
    # The published accuracy must be whatever the shipped metrics file says,
    # not a hard-coded number that drifts when the model is retrained.
    published = json.loads((Path(__file__).resolve().parents[1] / "models" / "metrics.json").read_text())
    assert payload["metrics"]["accuracy"] == pytest.approx(published["accuracy"], abs=1e-4)
    assert len(payload["groups"]) == 5


def test_predict_range_bands_and_ordering():
    for row in SAMPLE_ROWS:
        response = client.post("/api/predict", json={"values": row})
        assert response.status_code == 200
        payload = response.get_json()
        assert 0.0 <= payload["probability"] <= 1.0
        band = payload["band"]
        if payload["probability"] >= 0.6:
            assert band == "High risk"
        elif payload["probability"] >= 0.3:
            assert band == "Moderate risk"
        else:
            assert band == "Low risk"
        drivers = payload["drivers"]
        assert len(drivers) == 24
        magnitudes = [abs(driver["contribution"]) for driver in drivers]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert len(payload["explain"]["ups"]) <= 3
        assert len(payload["explain"]["downs"]) <= 3


def test_encoder_matches_src_features():
    """The web encoder must produce the same vector as the training pipeline."""
    import pandas as pd

    from src.features import encode

    for row in SAMPLE_ROWS:
        expected = encode(pd.DataFrame([row])).iloc[0][FEATURE_ORDER].tolist()
        values = encode_values(row)
        actual = [values[name] for name in FEATURE_ORDER]
        assert actual == expected


def test_probabilities_match_sklearn_wrapper():
    """score() must agree with XGBClassifier.predict_proba to float precision."""
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(ROOT / "models" / "model.json")

    matrix = np.array(
        [[encode_values(row)[name] for name in FEATURE_ORDER] for row in SAMPLE_ROWS],
        dtype=float,
    )
    expected = model.predict_proba(matrix)[:, 1]
    actual, contributions, base = score(matrix)
    assert np.allclose(actual, expected, atol=1e-6)
    # Contributions + bias must sum to the margin that produces the probability.
    margins = contributions.sum(axis=1) + base
    assert np.allclose(1.0 / (1.0 + np.exp(-margins)), expected, atol=1e-6)


def test_batch_scoring_and_counts():
    response = client.post("/api/batch", json={"rows": SAMPLE_ROWS})
    assert response.status_code == 200
    payload = response.get_json()
    results = payload["results"]
    assert len(results) == 3
    probabilities = [row["probability"] for row in results]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(payload["counts"].values()) == 3
    assert all(row["top_factor"] in FEATURE_LABELS.values() for row in results)


def test_batch_validation():
    assert client.post("/api/batch", json={"rows": []}).status_code == 400


def test_defaults_encode_without_any_fields():
    values = encode_values({})
    assert len(values) == 24
    # Numeric defaults come straight from the schema.
    for name, (_label, _low, _high, default) in NUMERIC_FEATURES.items():
        assert values[name] == default
