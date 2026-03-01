# Telco Customer Churn — Data Insights + ML Dashboard

An end-to-end churn prediction project built as a deployable Streamlit app:
- **Data Insights mode**: interactive churn analysis + business segments
- **Predict mode**: churn probability, thresholding, explanations, and batch scoring

## Tech Stack
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly

## Repo Structure
- `app/` Streamlit dashboard
- `src/` data loading, features, training, inference utilities
- `notebooks/` EDA + experimentation
- `models/` saved models
- `reports/figures/` exported charts

## Quickstart
```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
streamlit run app/Home.py
```
---