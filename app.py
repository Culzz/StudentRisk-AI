"""StudentRisk AI — instant academic-risk prediction with SHAP explanations."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
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
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "model.json")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return model, metrics


@st.cache_resource
def load_explainer(_model: XGBClassifier) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


def collect_inputs() -> pd.DataFrame:
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
    if probability >= 0.6:
        return "High risk", "#c0392b"
    if probability >= 0.3:
        return "Moderate risk", "#d68910"
    return "Low risk", "#1e8449"


def waterfall_chart(explainer: shap.TreeExplainer, row: pd.DataFrame):
    values = explainer(row.rename(columns=FEATURE_LABELS))
    fig = plt.figure()
    shap.plots.waterfall(values[0], max_display=12, show=False)
    fig.set_size_inches(9, 6)
    plt.tight_layout()
    return fig, values


model, metrics = load_model()
explainer = load_explainer(model)

st.title("🎓 StudentRisk AI")
st.write(
    "Enter a student's academic and behavioural data to get an instant risk prediction, "
    "explained feature by feature with SHAP."
)

row = collect_inputs()
probability = float(model.predict_proba(row)[0, 1])
band, colour = risk_band(probability)

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
    fig, shap_values = waterfall_chart(explainer, row)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Red bars push the student towards being at risk, blue bars towards being on track. "
        "The baseline is the average student in the training data."
    )

contributions = (
    pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[name] for name in FEATURE_ORDER],
            "Value": row.iloc[0].values,
            "SHAP impact": shap_values.values[0],
        }
    )
    .assign(magnitude=lambda d: d["SHAP impact"].abs())
    .sort_values("magnitude", ascending=False)
    .drop(columns="magnitude")
    .head(8)
    .reset_index(drop=True)
)

with st.expander("Top contributing factors (table)"):
    st.dataframe(contributions, use_container_width=True)

st.caption(
    "Decision support only — predictions are statistical estimates and should never replace "
    "a teacher's judgement."
)
