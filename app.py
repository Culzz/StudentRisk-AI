"""StudentRisk AI — academic-risk prediction web service.

A Flask app that serves a vintage single-page frontend (``public/``) plus a JSON
API powered by the trained XGBoost model:

- ``GET  /api/health``   — liveness probe
- ``GET  /api/schema``   — feature definitions, presets, metrics (drives the UI)
- ``POST /api/predict``  — one student → probability, band, exact TreeSHAP breakdown
- ``POST /api/batch``    — a whole class → probability, band, strongest driver per row

Runs locally with ``python app.py`` and deploys to Vercel as-is (this file exports
the top-level ``app`` instance Vercel looks for).
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    BINARY_FEATURES,
    FEATURE_LABELS,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
)

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

#: Form layout: groups of fields, rendered in this order by the frontend.
FIELD_GROUPS = [
    {
        "id": "grades",
        "title": "Grades",
        "icon": "award",
        "note": "The strongest signal in the model.",
        "fields": ["G1", "G2"],
        "selects": [],
    },
    {
        "id": "profile",
        "title": "Student profile",
        "icon": "user",
        "note": "Background basics.",
        "fields": ["age", "Medu", "Fedu", "traveltime"],
        "selects": ["sex", "area"],
    },
    {
        "id": "study",
        "title": "Study & attendance",
        "icon": "book",
        "note": "Effort and continuity.",
        "fields": ["studytime", "failures", "absences"],
        "selects": [],
    },
    {
        "id": "life",
        "title": "Life & wellbeing",
        "icon": "heart",
        "note": "Balance away from the desk.",
        "fields": ["famrel", "freetime", "goout", "Dalc", "Walc", "health"],
        "selects": [],
    },
    {
        "id": "support",
        "title": "Support & goals",
        "icon": "shield",
        "note": "Safety nets and ambitions.",
        "fields": list(BINARY_FEATURES),
        "selects": [],
    },
]

SELECT_FIELDS = [
    {"id": "sex", "title": "Sex", "icon": "user", "options": ["Female", "Male"], "default": "Female"},
    {"id": "area", "title": "Home area", "icon": "map", "options": ["Urban", "Rural"], "default": "Urban"},
]

#: One-click example profiles (ported from the original app).
PRESETS = {
    "high": {
        "id": "high",
        "label": "High-risk example",
        "tone": "high",
        "values": {
            "G1": 6, "G2": 7, "failures": 2, "studytime": 1, "absences": 30,
            "goout": 5, "freetime": 4, "Dalc": 3, "Walc": 4, "health": 3, "famrel": 3,
            "age": 17, "Medu": 1, "Fedu": 1, "traveltime": 2,
            "schoolsup": 1, "famsup": 0, "paid": 0, "activities": 0,
            "higher": 0, "internet": 0, "romantic": 1,
            "sex": "Female", "area": "Urban",
        },
    },
    "moderate": {
        "id": "moderate",
        "label": "Moderate-risk example",
        "tone": "moderate",
        "values": {
            "G1": 10, "G2": 11, "failures": 0, "studytime": 2, "absences": 8,
            "goout": 3, "freetime": 3, "Dalc": 1, "Walc": 2, "health": 4, "famrel": 4,
            "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1,
            "schoolsup": 0, "famsup": 1, "paid": 0, "activities": 1,
            "higher": 1, "internet": 1, "romantic": 0,
            "sex": "Male", "area": "Urban",
        },
    },
    "low": {
        "id": "low",
        "label": "Low-risk example",
        "tone": "low",
        "values": {
            "G1": 16, "G2": 17, "failures": 0, "studytime": 3, "absences": 1,
            "goout": 2, "freetime": 3, "Dalc": 1, "Walc": 1, "health": 5, "famrel": 5,
            "age": 16, "Medu": 4, "Fedu": 4, "traveltime": 1,
            "schoolsup": 0, "famsup": 1, "paid": 1, "activities": 1,
            "higher": 1, "internet": 1, "romantic": 0,
            "sex": "Female", "area": "Urban",
        },
    },
}

BAND_GUIDE = {
    "High risk": "Probability ≥ 60%. In a real setting this student would merit a "
                 "check-in conversation and a review of support options.",
    "Moderate risk": "Probability 30–59%. Worth monitoring; small improvements in "
                     "grades or attendance typically move students out of this band.",
    "Low risk": "Probability < 30%. On track by the model's standards — keep the "
                "habits that are working.",
}

BINARY_DEFAULTS = {name: name in ("famsup", "higher", "internet") for name in BINARY_FEATURES}
MAX_BATCH_ROWS = 2000
MAX_BODY_BYTES = 4 * 1024 * 1024  # stay under Vercel's 4.5 MB request limit

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_model() -> tuple[xgb.Booster, dict]:
    """Load the trained booster and its metrics once per process."""
    booster = xgb.Booster()
    booster.load_model(str(ROOT / "models" / "model.json"))
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return booster, metrics


def _to_flag(value) -> int:
    """Coerce booleans, numbers, and CSV strings ('1'/'yes'/'true') to 0/1."""
    if isinstance(value, str):
        return 1 if value.strip().lower() in ("1", "true", "yes", "y") else 0
    return 1 if value else 0


def encode_values(raw: dict) -> dict:
    """Normalise a raw submission into the model's 24 encoded feature values."""
    values: dict[str, int] = {}

    for name, (_label, low, high, default) in NUMERIC_FEATURES.items():
        try:
            number = float(raw.get(name, default))
        except (TypeError, ValueError):
            number = float(default)
        values[name] = int(round(min(max(number, low), high)))

    for name in BINARY_FEATURES:
        values[name] = _to_flag(raw.get(name, BINARY_DEFAULTS[name]))

    # Accept either the human fields (sex/area) or the encoded ones directly.
    if raw.get("is_female") is not None:
        values["is_female"] = _to_flag(raw.get("is_female"))
    else:
        values["is_female"] = 1 if str(raw.get("sex", "Female")).strip().lower() in ("female", "f") else 0

    if raw.get("is_urban") is not None:
        values["is_urban"] = _to_flag(raw.get("is_urban"))
    else:
        values["is_urban"] = 1 if str(raw.get("area", raw.get("address", "Urban"))).strip().lower() in ("urban", "u") else 0

    return values


def risk_band(probability: float) -> tuple[str, str]:
    """Map a failure probability to a (label, tone) pair."""
    if probability >= 0.6:
        return "High risk", "high"
    if probability >= 0.3:
        return "Moderate risk", "moderate"
    return "Low risk", "low"


def score(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (probabilities, per-feature contributions, base value) for a matrix.

    Contributions come from XGBoost's own ``pred_contribs=True`` (exact TreeSHAP);
    they sum, with the bias column, to the raw margin the logistic link turns
    into a probability.
    """
    booster, _metrics = load_model()
    data = xgb.DMatrix(matrix, feature_names=FEATURE_ORDER)
    contribs = booster.predict(data, pred_contribs=True)          # (N, 25)
    margins = contribs.sum(axis=1)
    probabilities = 1.0 / (1.0 + np.exp(-margins))
    return probabilities, contribs[:, :-1], float(contribs[0, -1])


def build_driver_table(values: dict, contributions: np.ndarray) -> list[dict]:
    """Full feature breakdown, strongest influence first."""
    rows = []
    for index, name in enumerate(FEATURE_ORDER):
        contribution = float(contributions[index])
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS[name],
            "value": values[name],
            "contribution": round(contribution, 4),
            "direction": "up" if contribution > 0 else "down",
        })
    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return rows


def build_explanation(drivers: list[dict]) -> dict:
    """Plain-English summary: top three drivers up, top three in the student's favour."""
    ups = [row for row in drivers if row["contribution"] > 0][:3]
    downs = [row for row in drivers if row["contribution"] < 0][:3]
    return {
        "ups": [{"label": row["label"], "value": row["value"]} for row in ups],
        "downs": [{"label": row["label"], "value": row["value"]} for row in downs],
        "empty": not ups and not downs,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES


@app.get("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(PUBLIC, "index.html")


@app.get("/assets/<path:filename>")
def asset(filename: str):
    """Serve frontend assets (Vercel's CDN handles these in production)."""
    return send_from_directory(PUBLIC / "assets", filename)


@app.get("/favicon.svg")
def favicon():
    return send_from_directory(PUBLIC, "favicon.svg")


@app.get("/api/health")
def health():
    return jsonify(ok=True, service="studentrisk-ai", model="xgboost")


@app.get("/api/schema")
def schema():
    """Everything the frontend needs to build the form and model card."""
    _booster, metrics = load_model()
    numeric = {
        name: {
            "label": label,
            "low": low,
            "high": high,
            "default": default,
        }
        for name, (label, low, high, default) in NUMERIC_FEATURES.items()
    }
    binaries = {
        name: {"label": label, "default": BINARY_DEFAULTS[name]}
        for name, label in BINARY_FEATURES.items()
    }
    return jsonify(
        numeric=numeric,
        binaries=binaries,
        selects=SELECT_FIELDS,
        groups=FIELD_GROUPS,
        presets=PRESETS,
        band_guide=BAND_GUIDE,
        bands={"high": 0.6, "moderate": 0.3},
        feature_order=FEATURE_ORDER,
        feature_labels=FEATURE_LABELS,
        metrics=metrics,
    )


@app.post("/api/predict")
def predict():
    """Score one student and return the full explanation."""
    payload = request.get_json(silent=True) or {}
    values = encode_values(payload.get("values") or payload)

    matrix = np.array([[values[name] for name in FEATURE_ORDER]], dtype=float)
    probabilities, contributions, base_value = score(matrix)

    probability = float(probabilities[0])
    band, tone = risk_band(probability)
    drivers = build_driver_table(values, contributions[0])

    return jsonify(
        probability=round(probability, 4),
        band=band,
        tone=tone,
        base_value=round(base_value, 4),
        drivers=drivers,
        explain=build_explanation(drivers),
    )


@app.post("/api/batch")
def batch():
    """Score a whole class: rows are dicts of raw feature values."""
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return jsonify(error="`rows` must be a non-empty list of student objects."), 400
    if len(rows) > MAX_BATCH_ROWS:
        return jsonify(error=f"Too many rows (max {MAX_BATCH_ROWS})."), 400

    encoded = [encode_values(row) for row in rows]
    matrix = np.array([[values[name] for name in FEATURE_ORDER] for values in encoded], dtype=float)
    probabilities, contributions, _base = score(matrix)

    results = []
    for index, probability in enumerate(probabilities):
        band, tone = risk_band(float(probability))
        top_index = int(np.abs(contributions[index]).argmax())
        results.append({
            "student": rows[index].get("student") or f"S{index + 1:03d}",
            "probability": round(float(probability), 4),
            "band": band,
            "tone": tone,
            "top_factor": FEATURE_LABELS[FEATURE_ORDER[top_index]],
        })
    results.sort(key=lambda row: row["probability"], reverse=True)

    counts = {"high": 0, "moderate": 0, "low": 0}
    for row in results:
        counts[row["tone"]] += 1
    return jsonify(results=results, counts=counts)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Payload too large — keep class uploads under 4 MB."), 413


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
