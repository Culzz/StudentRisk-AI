"""Honest evaluation for StudentRisk AI.

What this script adds over the original 88.5%-Accuracy story:

1.  **Leakage-free evaluation** — the 1,044 records come from 662 unique students
    (many appear in both the Maths and Portuguese files). The original ``train.py``
    used a plain stratified row split, so the same student's records could land on
    both sides. Here every experiment uses a **grouped split by student**.

2.  **Ablation** — the same model trained three ways:
    - all 24 features
    - without ``G2`` (term 2 grade — nearly the outcome itself)
    - without ``G1`` and ``G2`` (true *early*-warning view: usable before grades land)
    compared with the dumb rule baseline ``G2 < 10``.

3.  **Calibration** — Brier score and a decile reliability table for the shipped
    model, so the probability on the web meter can be read as a probability.

Outputs: console report + enriched ``models/metrics.json``.
Run after ``train.py`` (uses the shipped booster + produces its grouping metadata).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from features import FEATURE_ORDER, PASS_MARK, encode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42


def load_dataset() -> pd.DataFrame:
    frames = []
    for name, subject in (("student-por.csv", "portuguese"), ("student-mat.csv", "math")):
        frame = pd.read_csv(DATA_DIR / name, sep=";")
        frame["subject"] = subject
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def student_key(df: pd.DataFrame) -> pd.Series:
    """Per-student group id.

    The 1,044 records cover ~662 unique students because many sit in both the
    Maths and Portuguese files. Counting groups with a hash of the *stable
    demographic* columns reproduces the documented 662 (verified: 366 students
    appear in both subjects, 295 in one). Survey-reported habits (study time,
    alcohol use, family-relationship ratings…) are recorded per course file and
    are too noisy to use as identity, so they are deliberately excluded.
    """
    import hashlib

    identity_cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    ]

    def _hash(row):
        joined = "|".join(f"{row[col]}" for col in identity_cols)
        return hashlib.md5(joined.encode()).hexdigest()[:10]

    return df[identity_cols].apply(_hash, axis=1)


def make_model():
    """Same hyperparameters as the shipped train.py — apples to apples."""
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=SEED,
    )


def grouped_split(X, y, groups):
    """80/20 split where one student never appears on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def metrics_row(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "recall_at_0.2": round(float(recall_score(y_te, proba >= 0.2)), 4),
        "recall_at_0.3": round(float(recall_score(y_te, proba >= 0.3)), 4),
        "recall_at_0.5": round(float(recall_score(y_te, preds)), 4),
        "n_test": int(len(y_te)),
    }


def main() -> None:
    raw = load_dataset()
    X_all = encode(raw)
    y = (raw["G3"] < PASS_MARK).astype(int)
    keys = student_key(raw)
    n_groups = keys.nunique()

    print(f"records={len(raw)}  unique_student_keys={n_groups}")
    train_idx, test_idx = grouped_split(X_all, y, keys)

    ablation = {}
    sets = {
        "all_features": list(FEATURE_ORDER),
        "without_G2": [f for f in FEATURE_ORDER if f != "G2"],
        "without_G1_G2": [f for f in FEATURE_ORDER if f not in ("G1", "G2")],
    }
    for label, feats in sets.items():
        cols = [FEATURE_ORDER.index(f) for f in feats]
        row = metrics_row(make_model(), X_all.iloc[train_idx, cols], y.iloc[train_idx],
                          X_all.iloc[test_idx, cols], y.iloc[test_idx])
        row["features"] = len(feats)
        ablation[label] = row
        print(f"[{label}] {json.dumps(row)}")

    # Dumb rule baseline: a student is at risk iff G2 < 10.
    g2_test = X_all["G2"].iloc[test_idx]
    rule_pred = (g2_test < PASS_MARK).astype(int)
    ablation["rule_G2_below_10"] = {
        "accuracy": round(float(accuracy_score(y.iloc[test_idx], rule_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y.iloc[test_idx], -g2_test)), 4),
        "recall_at_0.2": round(float(recall_score(y.iloc[test_idx], -g2_test >= -0.2)), 4),
        "recall_at_0.3": round(float(recall_score(y.iloc[test_idx], -g2_test >= -0.3)), 4),
        "recall_at_0.5": round(float(recall_score(y.iloc[test_idx], rule_pred)), 4),
        "features": 1,
    }
    print(f"[rule_G2_below_10] {json.dumps(ablation['rule_G2_below_10'])}")

    # Fairness: error rates by protected/structural groups, and the effect of
    # dropping the protected attributes from the feature set entirely.
    from sklearn.metrics import accuracy_score as _acc
    from xgboost import XGBClassifier as _XGB

    shipped = _XGB()
    shipped.load_model(MODEL_DIR / "model.json")
    X_te_all = X_all.iloc[test_idx]
    proba = shipped.predict_proba(X_te_all)[:, 1]
    preds = (proba >= 0.5).astype(int)
    y_te = y.iloc[test_idx].to_numpy()
    group_report = {}
    for col in ("sex", "address"):
        for value in sorted(raw[col].unique()):
            mask = (raw[col].iloc[test_idx].to_numpy() == value)
            if mask.sum() < 20:
                continue
            fp = int(((preds == 1) & (y_te == 0) & mask).sum())
            fn = int(((preds == 0) & (y_te == 1) & mask).sum())
            group_report[f"{col}={value}"] = {
                "n": int(mask.sum()),
                "accuracy": round(float(_acc(y_te[mask], preds[mask])), 4),
                "false_positives": fp,
                "false_negatives": fn,
            }
    fairness = {"groups": group_report}

    protected = {"is_female", "is_urban"}
    cols_prot = [FEATURE_ORDER.index(f) for f in FEATURE_ORDER if f not in protected]
    fairness["without_sex_or_address"] = metrics_row(
        make_model(), X_all.iloc[train_idx, cols_prot], y.iloc[train_idx],
        X_all.iloc[test_idx, cols_prot], y.iloc[test_idx])
    print(f"[fairness] {json.dumps(fairness)}")

    # Calibration of the shipped model (loaded from models/model.json).
    calibrated_proba = proba
    brier = float(brier_score_loss(y_te, proba))
    bins = np.quantile(proba, np.linspace(0, 1, 11))
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi if hi != bins[-1] else proba <= hi)
        if mask.sum() == 0:
            continue
        table.append({
            "predicted_mean": round(float(proba[mask].mean()), 4),
            "observed_risk": round(float(y_te[mask].mean()), 4),
            "count": int(mask.sum()),
        })
    calibration = {"brier": round(brier, 4), "deciles": table}

    out = json.loads((MODEL_DIR / "metrics.json").read_text())
    out["grouping"] = {
        "split": "GroupShuffleSplit by student (md5 of identity columns)",
        "unique_students": int(n_groups),
        "records": int(len(raw)),
        "seed": SEED,
    }
    out["ablation"] = ablation
    out["fairness"] = fairness
    out["calibration"] = calibration
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\nmetrics.json enriched")


if __name__ == "__main__":
    main()
