"""Feature schema shared by training and the web app."""

from __future__ import annotations

BINARY_FEATURES = {
    "schoolsup": "Extra school support",
    "famsup": "Family educational support",
    "paid": "Paid extra classes",
    "activities": "Extra-curricular activities",
    "higher": "Wants higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
}

NUMERIC_FEATURES = {
    "age": ("Age", 15, 22, 17),
    "Medu": ("Mother's education (0-4)", 0, 4, 2),
    "Fedu": ("Father's education (0-4)", 0, 4, 2),
    "traveltime": ("Travel time to school (1-4)", 1, 4, 1),
    "studytime": ("Weekly study time (1-4)", 1, 4, 2),
    "failures": ("Past class failures", 0, 4, 0),
    "famrel": ("Family relationship quality (1-5)", 1, 5, 4),
    "freetime": ("Free time after school (1-5)", 1, 5, 3),
    "goout": ("Going out with friends (1-5)", 1, 5, 3),
    "Dalc": ("Workday alcohol use (1-5)", 1, 5, 1),
    "Walc": ("Weekend alcohol use (1-5)", 1, 5, 2),
    "health": ("Current health status (1-5)", 1, 5, 4),
    "absences": ("Absences this year", 0, 60, 4),
    "G1": ("Term 1 grade (0-20)", 0, 20, 11),
    "G2": ("Term 2 grade (0-20)", 0, 20, 11),
}

DERIVED_FEATURES = {
    "is_female": "Female",
    "is_urban": "Lives in an urban area",
}

FEATURE_ORDER = (
    list(NUMERIC_FEATURES) + list(BINARY_FEATURES) + list(DERIVED_FEATURES)
)

FEATURE_LABELS = {
    **{name: label for name, (label, *_rest) in NUMERIC_FEATURES.items()},
    **BINARY_FEATURES,
    **DERIVED_FEATURES,
}

PASS_MARK = 10


def encode(raw):
    """Turn raw UCI-style columns into the numeric matrix the model expects."""
    import pandas as pd  # imported lazily so the web API can use the schema without pandas

    df = pd.DataFrame(index=raw.index)
    for name in NUMERIC_FEATURES:
        df[name] = pd.to_numeric(raw[name])
    for name in BINARY_FEATURES:
        df[name] = raw[name].map({"yes": 1, "no": 0}).astype(int)
    df["is_female"] = raw["sex"].map({"F": 1, "M": 0}).astype(int)
    df["is_urban"] = raw["address"].map({"U": 1, "R": 0}).astype(int)
    return df[FEATURE_ORDER]
