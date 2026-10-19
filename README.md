# 🎓 StudentRisk AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-1f77b4.svg)](https://xgboost.readthedocs.io/)

A live Streamlit web app: enter a student's behavioural and academic data and get an instant
risk-of-failing prediction, with a SHAP waterfall chart showing exactly which factors drove it.

## Features

- **Instant risk prediction** — 24 behavioural and academic features in, one probability out
  (Low / Moderate / High band with colour-coded status).
- **Exact SHAP explanations** — every prediction comes with a ranked bar chart of the factors
  that pushed it up or down, in log-odds, from the average student's baseline.
- **No heavy runtime** — TreeSHAP values are computed through XGBoost's own
  `pred_contribs=True`, so the compiled `shap` extension is not needed at deploy time.
- **One-click deploy** — runs on Streamlit Community Cloud with no secrets or configuration.
- **Reproducible training** — `src/train.py` regenerates `models/model.json` and
  `models/metrics.json` from the bundled UCI data with a fixed seed.

## Model

- Data: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
  (1,044 course records — 395 Maths + 649 Portuguese — from 662 unique students).
- Target: final grade `G3 < 10` (failing the year).
- 24 features: term grades, study time, past failures, absences, going out, alcohol use,
  health, family support, and background.
- Split: stratified 80/20, `random_state=42`.

| Metric | Value |
|---|---|
| Test accuracy | **88.5%** |
| ROC-AUC | **0.9415** |
| Train / test size | 835 / 209 |
| At-risk rate | 22.0% |

Exact values live in [`models/metrics.json`](models/metrics.json); retraining reproduces them.

## Run locally

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt   # requirements.txt alone is enough to run the app
python src/train.py                   # optional: regenerates models/model.json
streamlit run app.py
```

No API keys, database, or configuration are required — the model and metrics ship with the repo.

## Deploy free on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo (`models/model.json` must be committed).
2. Go to https://share.streamlit.io → **Create app** → pick the repo/branch.
3. Main file path: `app.py`. Any Python 3.10+ works.
4. Deploy — you get a public URL like `https://<app-name>.streamlit.app`.

Dependencies come from `requirements.txt`; no secrets or config are required.

## Layout

```
app.py               Streamlit UI + SHAP explanation
src/features.py      Shared feature schema and encoding
src/train.py         Training script
models/              Trained model (model.json) + metrics (metrics.json)
data/                UCI CSVs + attribute descriptions
.devcontainer/       Codespaces / VS Code dev container config
```

## Limitations & Ethics

- The model is trained on Portuguese secondary-school data (2005–2006); treat performance on
  other populations as unvalidated until retrained locally.
- Predictions are statistical estimates for **decision support only** — they must never replace
  a teacher's judgement, and should not be used as the sole basis for any decision about a
  student.

## Citation

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5TG7T

## License

Released under the [MIT License](LICENSE).

## Author

**Omeche Chimaobi Benedict** — [github.com/Culzz](https://github.com/Culzz) · Omechechima@gmail.com
