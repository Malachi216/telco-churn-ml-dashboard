import json
from pathlib import Path

def load_meta(name: str):
    p = Path("models") / f"telco_churn_{name}_meta.json"
    return json.loads(p.read_text())

log = load_meta("logreg")
xgb = load_meta("xgb")

readme = f"""# Telco Customer Churn — Data Insights + ML Dashboard

A deployable end-to-end churn prediction project built with **Streamlit**:
- **Data Insights**: interactive churn analysis, segmentation, revenue-at-risk
- **Predict Churn**: batch scoring + threshold tuning + model evaluation

## Why this project
Customer churn directly impacts revenue. This project shows the full pipeline from raw data → analysis → predictive model → deployed dashboard.

## Models
| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | {log['roc_auc']:.3f} | {log['pr_auc']:.3f} |
| XGBoost | {xgb['roc_auc']:.3f} | {xgb['pr_auc']:.3f} |

## Dashboard Features
### Data Insights
- KPI cards (churn rate, tenure, monthly charges, revenue-at-risk)
- Churn breakdown by contract, payment method, internet service
- Tenure and charges distributions by churn

### Predict Churn
- Model selector (LogReg / XGB)
- Threshold slider + live precision/recall/accuracy on test set
- Batch scoring from uploaded CSV + downloadable results

## Repo Structure
- `app/` Streamlit dashboard
- `src/utils/` data loading + cleaning
- `src/models/` training + inference
- `models/` saved models + metadata
- `data/raw/` dataset (ignored from git)

## Quickstart
```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
python -m src.models.train
streamlit run app/Home.py