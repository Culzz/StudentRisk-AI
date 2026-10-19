"""StudentRisk AI — instant academic-risk prediction with SHAP explanations."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from features import (  # noqa: E402
    BINARY_FEATURES,
    FEATURE_LABELS,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
)

MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="StudentRisk AI", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model() -> tuple[XGBClassifier, dict]:
    """Load the trained XGBoost model and its metrics (cached across reruns)."""
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return model, metrics


def shap_contributions(model: XGBClassifier, row: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values from XGBoost itself (last column is the base value)."""
    matrix = xgb.DMatrix(row.values, feature_names=FEATURE_ORDER)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)[0]
    return contributions[:-1], float(contributions[-1])


def collect_inputs() -> pd.DataFrame:
    """Build the sidebar of sliders/selectors and return one encoded row."""
    values: dict[str, int] = {}
    with st.sidebar:
        st.header("Student profile")
        st.caption("Academic")
        for name in ("G1", "G2", "failures", "studytime", "absences"):
            label, low, high, default = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, default)

        st.caption("Behaviour & wellbeing")
        for name in ("goout", "freetime", "Dalc", "Walc", "health", "famrel"):
            label, low, high, default = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, default)

        st.caption("Background")
        for name in ("age", "Medu", "Fedu", "traveltime"):
            label, low, high, default = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, default)

        values["is_female"] = int(st.selectbox("Sex", ("Female", "Male")) == "Female")
        values["is_urban"] = int(st.selectbox("Home area", ("Urban", "Rural")) == "Urban")
        for name, label in BINARY_FEATURES.items():
            values[name] = int(st.checkbox(label, value=name in ("famsup", "higher", "internet")))

    return pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def risk_band(probability: float) -> tuple[str, str]:
    """Map a failure probability to a (label, colour) band for display."""
    if probability >= 0.6:
        return "High risk", "#c0392b"
    if probability >= 0.3:
        return "Moderate risk", "#d68910"
    return "Low risk", "#1e8449"


def contribution_chart(table: pd.DataFrame):
    """Horizontal bar chart of SHAP impacts — red pushes risk up, blue down."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = table.iloc[::-1]
    colours = ["#c0392b" if value > 0 else "#2874c9" for value in ordered["SHAP impact"]]
    ax.barh(ordered["Feature"] + "  =  " + ordered["Value"].astype(str), ordered["SHAP impact"], color=colours)
    ax.axvline(0, color="#444", linewidth=1)
    ax.set_xlabel("Impact on risk (log-odds)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


model, metrics = load_model()

st.title("🎓 StudentRisk AI")
st.write(
    "Enter a student's academic and behavioural data to get an instant risk prediction, "
    "explained feature by feature with SHAP."
)

row = collect_inputs()
probability = float(model.predict_proba(row)[0, 1])
band, colour = risk_band(probability)
values, base_value = shap_contributions(model, row)

table = (
    pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[name] for name in FEATURE_ORDER],
            "Value": row.iloc[0].values,
            "SHAP impact": values,
        }
    )
    .assign(magnitude=lambda d: d["SHAP impact"].abs())
    .sort_values("magnitude", ascending=False)
    .drop(columns="magnitude")
    .reset_index(drop=True)
)

left, right = st.columns([1, 2])
with left:
    st.metric("Probability of failing the year", f"{probability:.0%}")
    st.markdown(
        f"<div style='padding:0.6rem 1rem;border-radius:8px;background:{colour};"
        f"color:white;font-weight:600;text-align:center'>{band}</div>",
        unsafe_allow_html=True,
    )
    st.progress(probability)
    st.caption(
        f"Model: XGBoost · test accuracy {metrics['accuracy']:.0%} · "
        f"ROC-AUC {metrics['roc_auc']:.2f} · trained on {metrics['n_train']} students "
        "(UCI Student Performance)."
    )

with right:
    st.subheader("Why this prediction?")
    st.pyplot(contribution_chart(table.head(12)), clear_figure=True)
    st.caption(
        "Exact TreeSHAP values. Red bars push the student towards being at risk, blue bars "
        f"towards being on track, starting from the average student (base value {base_value:.2f})."
    )

with st.expander("Top contributing factors (table)"):
    st.dataframe(table.head(10), use_container_width=True)

st.caption(
    "Decision support only — predictions are statistical estimates and should never replace "
    "a teacher's judgement."
)
