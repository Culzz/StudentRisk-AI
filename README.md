# 🎓 StudentRisk AI

A live Streamlit web app: enter a student's behavioural and academic data and get an instant
risk-of-failing prediction, with a SHAP waterfall chart showing exactly which factors drove it.

**Stack:** Python · XGBoost · SHAP · pandas · scikit-learn · Streamlit (Streamlit Community Cloud)

## Model

- Data: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
  (1,044 students, Portuguese + Maths courses).
- Target: final grade `G3 < 10` (failing the year).
- 24 features: term grades, study time, past failures, absences, going out, alcohol use,
  health, family support, and background.
- Test performance: **~89% accuracy, 0.94 ROC-AUC** (see `models/metrics.json`).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py     # optional: regenerates models/model.json
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo (`models/model.json` must be committed).
2. Go to https://share.streamlit.io → **Create app** → pick the repo/branch.
3. Main file path: `app.py`. Python version: 3.10+.
4. Deploy — you get a public URL like `https://<app-name>.streamlit.app`.

Dependencies come from `requirements.txt`; no secrets or config are required.

## Layout

```
app.py               Streamlit UI + SHAP explanation
src/features.py      Shared feature schema and encoding
src/train.py         Training script
models/              Trained model + metrics
data/                UCI CSVs
```

Decision support only — predictions are statistical estimates, not a substitute for a teacher's
judgement.
