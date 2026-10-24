"""Train the StudentRisk AI model on the UCI Student Performance dataset."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from features import FEATURE_ORDER, PASS_MARK, encode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def load_dataset() -> pd.DataFrame:
    """Load and concatenate the Maths and Portuguese UCI CSVs, tagged by subject."""
    frames = []
    for name, subject in (("student-por.csv", "portuguese"), ("student-mat.csv", "math")):
        frame = pd.read_csv(DATA_DIR / name, sep=";")
        frame["subject"] = subject
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def student_key(df: pd.DataFrame) -> pd.Series:
    """Per-student group id: the 1,044 records cover ~662 unique students because
    many appear in both the Maths and Portuguese files. A row split would leak
    the same student into train and test; a hash of the stable *demographic*
    columns reproduces the documented 662 identities (366 students attend both
    course files, 295 just one). Survey-reported habits are recorded per course
    and are too noisy for identity, so they are excluded."""
    import hashlib

    identity_cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    ]

    def _hash(row):
        joined = "|".join(f"{row[col]}" for col in identity_cols)
        return hashlib.md5(joined.encode()).hexdigest()[:10]

    return df[identity_cols].apply(_hash, axis=1)


def main() -> None:
    """Train the classifier, evaluate on the held-out split, and save model + metrics."""
    raw = load_dataset()
    X = encode(raw)
    y = (raw["G3"] < PASS_MARK).astype(int)
    keys = student_key(raw)
    n_students = int(keys.nunique())

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, keys))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "at_risk_rate": round(float(y.mean()), 4),
        "n_students": n_students,
        "split": "grouped by student (GroupShuffleSplit, seed 42)",
        "features": FEATURE_ORDER,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "model.json")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(classification_report(y_test, preds, target_names=["on track", "at risk"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
