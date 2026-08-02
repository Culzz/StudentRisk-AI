"""Train the StudentRisk AI model on the UCI Student Performance dataset."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FEATURE_ORDER, PASS_MARK, encode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def load_dataset() -> pd.DataFrame:
    frames = []
    for name, subject in (("student-por.csv", "portuguese"), ("student-mat.csv", "math")):
        frame = pd.read_csv(DATA_DIR / name, sep=";")
        frame["subject"] = subject
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    raw = load_dataset()
    X = encode(raw)
    y = (raw["G3"] < PASS_MARK).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
        "features": FEATURE_ORDER,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "model.json")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(classification_report(y_test, preds, target_names=["on track", "at risk"]))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
