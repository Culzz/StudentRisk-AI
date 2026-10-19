"""StudentRisk AI — instant academic-risk prediction with SHAP explanations."""

import io
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

# One-click example profiles so visitors can try the app without filling 24 widgets.
PRESETS = {
    "🔴 High-risk example": {
        "G1": 6, "G2": 7, "failures": 2, "studytime": 1, "absences": 30,
        "goout": 5, "freetime": 4, "Dalc": 3, "Walc": 4, "health": 3, "famrel": 3,
        "age": 17, "Medu": 1, "Fedu": 1, "traveltime": 2,
        "schoolsup": 1, "famsup": 0, "paid": 0, "activities": 0,
        "higher": 0, "internet": 0, "romantic": 1,
        "sex_choice": "Female", "area_choice": "Urban",
    },
    "🟡 Moderate-risk example": {
        "G1": 10, "G2": 11, "failures": 0, "studytime": 2, "absences": 8,
        "goout": 3, "freetime": 3, "Dalc": 1, "Walc": 2, "health": 4, "famrel": 4,
        "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1,
        "schoolsup": 0, "famsup": 1, "paid": 0, "activities": 1,
        "higher": 1, "internet": 1, "romantic": 0,
        "sex_choice": "Male", "area_choice": "Urban",
    },
    "🟢 Low-risk example": {
        "G1": 16, "G2": 17, "failures": 0, "studytime": 3, "absences": 1,
        "goout": 2, "freetime": 3, "Dalc": 1, "Walc": 1, "health": 5, "famrel": 5,
        "age": 16, "Medu": 4, "Fedu": 4, "traveltime": 1,
        "schoolsup": 0, "famsup": 1, "paid": 1, "activities": 1,
        "higher": 1, "internet": 1, "romantic": 0,
        "sex_choice": "Female", "area_choice": "Urban",
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


def apply_preset_buttons() -> bool:
    """Render example-profile buttons; apply and rerun when one is clicked."""
    st.sidebar.subheader("Load an example")
    clicked = False
    for name, values in PRESETS.items():
        if st.sidebar.button(name, width='stretch'):
            for key, value in values.items():
                st.session_state[key] = value
            st.session_state["preset"] = name
            clicked = True
    if clicked:
        st.rerun()
    return clicked


def collect_inputs() -> pd.DataFrame:
    """Build the sidebar of sliders/selectors and return one encoded row."""
    apply_preset_buttons()

    values: dict[str, int] = {}

    with st.sidebar:
        st.header("Student profile")
        for name in ("G1", "G2", "failures", "studytime", "absences"):
            label, low, high, _ = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, key=name)

        st.caption("Behaviour & wellbeing")
        for name in ("goout", "freetime", "Dalc", "Walc", "health", "famrel"):
            label, low, high, _ = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, key=name)

        st.caption("Background")
        for name in ("age", "Medu", "Fedu", "traveltime"):
            label, low, high, _ = NUMERIC_FEATURES[name]
            values[name] = st.slider(label, low, high, key=name)

        st.session_state.setdefault("sex_choice", "Female")
        st.session_state.setdefault("area_choice", "Urban")
        values["is_female"] = int(st.selectbox("Sex", ("Female", "Male"),
                                               key="sex_choice") == "Female")
        values["is_urban"] = int(st.selectbox("Home area", ("Urban", "Rural"),
                                              key="area_choice") == "Urban")
        for name, label in BINARY_FEATURES.items():
            st.session_state.setdefault(name, name in ("famsup", "higher", "internet"))
            values[name] = int(st.checkbox(label, key=name))

    return pd.DataFrame([[values[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def collect_batch() -> pd.DataFrame:
    """Sidebar uploader for whole-class scoring; returns the encoded matrix."""
    with st.sidebar:
        st.header("Whole class")
        st.caption(f"Upload a CSV with these {len(FEATURE_ORDER)} columns: "
                   f"`{', '.join(FEATURE_ORDER[:6])}`, …")
        uploaded = st.file_uploader("Class CSV", type="csv")
    if uploaded is None:
        return pd.DataFrame(columns=FEATURE_ORDER)
    raw = pd.read_csv(uploaded)
    missing = [c for c in FEATURE_ORDER if c not in raw.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return pd.DataFrame(columns=FEATURE_ORDER)
    return raw[FEATURE_ORDER]


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


def plain_language(table: pd.DataFrame) -> str:
    """Turn the top SHAP drivers into readable sentences for non-ML readers."""
    ups = table[table["SHAP impact"] > 0].head(3)
    downs = table[table["SHAP impact"] < 0].tail(3).iloc[::-1]
    lines = ["**In plain English:**"]
    for _, row in ups.iterrows():
        lines.append(f"- 📈 **{row['Feature']}** (recorded: {row['Value']}) pushed the risk **up** the most.")
    if not ups.empty:
        lines.append("")
    for _, row in downs.iterrows():
        lines.append(f"- 📉 **{row['Feature']}** (recorded: {row['Value']}) worked **in the student's favour**.")
    if downs.empty and ups.empty:
        lines.append("- No single factor dominated this prediction.")
    return "\n".join(lines)


def score_batch(model: XGBClassifier, matrix: pd.DataFrame) -> pd.DataFrame:
    """Score a whole class: probability, band, and top contributing factor per row."""
    proba = model.predict_proba(matrix)[:, 1]
    contribs = model.get_booster().predict(
        xgb.DMatrix(matrix.values, feature_names=FEATURE_ORDER), pred_contribs=True)
    top_idx = np.abs(contribs[:, :-1]).argmax(axis=1)
    top_names = [FEATURE_ORDER[i] for i in top_idx]
    bands = [risk_band(p)[0] for p in proba]
    out = matrix.copy()
    out.insert(0, "Student", [f"S{i + 1:03d}" for i in range(len(matrix))])
    out["risk_probability"] = proba.round(3)
    out["risk_band"] = bands
    out["top_factor"] = [FEATURE_LABELS[n] for n in top_names]
    return out.sort_values("risk_probability", ascending=False)


model, metrics = load_model()

st.title("🎓 StudentRisk AI")
st.write(
    "Enter a student's academic and behavioural data to get an instant risk prediction, "
    "explained feature by feature with SHAP."
)

mode = st.sidebar.radio("Mode", ("Single student", "Whole class (CSV)"))

if mode == "Whole class (CSV)":
    matrix = collect_batch()
    if matrix.empty:
        st.info("Upload a class CSV in the sidebar to score everyone at once. "
                "Tip: run the app with the single-student mode first to see the expected columns.")
        st.stop()

    scored = score_batch(model, matrix)
    st.subheader(f"Class risk report — {len(scored)} students")
    st.dataframe(scored, width='stretch')
    st.download_button(
        "⬇️ Download class report (CSV)",
        scored.to_csv(index=False), file_name="class_risk_report.csv", mime="text/csv",
    )
    counts = scored["risk_band"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High risk", int(counts.get("High risk", 0)))
    c2.metric("🟡 Moderate risk", int(counts.get("Moderate risk", 0)))
    c3.metric("🟢 Low risk", int(counts.get("Low risk", 0)))
    st.caption("Each row includes the single strongest driver of that student's score.")
    st.stop()

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
    st.markdown(plain_language(table))
    st.caption(
        f"Model: XGBoost · test accuracy {metrics['accuracy']:.0%} · "
        f"ROC-AUC {metrics['roc_auc']:.2f} · trained on {metrics['n_train']} students "
        "(UCI Student Performance)."
    )

with right:
    st.subheader("Why this prediction?")
    fig = contribution_chart(table.head(12))
    st.pyplot(fig, clear_figure=False)
    st.caption(
        "Exact TreeSHAP values. Red bars push the student towards being at risk, blue bars "
        f"towards being on track, starting from the average student (base value {base_value:.2f})."
    )
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.download_button(
        "⬇️ Download explanation (PNG)", png.getvalue(),
        file_name="risk_explanation.png", mime="image/png",
    )

with st.expander("Top contributing factors (table)"):
    st.dataframe(table.head(10), width='stretch')

with st.expander("What do the risk bands mean?"):
    for label, text in BAND_GUIDE.items():
        st.markdown(f"**{label}** — {text}")

st.caption(
    "Decision support only — predictions are statistical estimates and should never replace "
    "a teacher's judgement."
)
