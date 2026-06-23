
# Telco Customer Churn — Data Insights + ML Dashboard

A deployable end-to-end churn prediction project built with **Streamlit**.

This project demonstrates an **end-to-end data science workflow**:
- Data cleaning and preprocessing
- Exploratory analysis and business insights
- Predictive modeling
- Model evaluation
- Deployment through an interactive dashboard

---

# Business Problem

Customer churn is one of the biggest drivers of revenue loss in subscription businesses.

The goal of this project is to:

• Identify patterns associated with churn  
• Predict which customers are most likely to leave  
• Provide actionable insights through interactive analytics

---

# Models

| Model | ROC-AUC | PR-AUC |
|------|--------|--------|
| Logistic Regression | 0.842 | 0.632 |
| XGBoost | 0.841 | 0.656 |

Both models are saved and available inside the Streamlit dashboard.

---

# Dashboard Features

## Data Insights
Interactive analytics including:

• KPI cards (churn rate, average tenure, revenue at risk)  
• Churn breakdown by contract type  
• Churn by internet service  
• Churn by payment method  
• Tenure and monthly charge distributions  

## Predict Churn

• Select model (Logistic Regression or XGBoost)  
• Adjustable decision threshold  
• Precision / Recall / Accuracy metrics  
• Confusion matrix  
• Batch scoring via CSV upload  
• Download predictions

---

# Project Structure

```
telco-churn-ml-dashboard
│
├── app/ # Streamlit dashboard
├── src/
│ ├── utils/ # Data loading and cleaning
│ ├── models/ # Training + inference
│
├── models/ # Saved models + metadata
├── data/raw/ # Dataset (not committed)
├── notebooks/ # EDA / experiments
├─ README.md
├─ reports
│  └─ figures
│     ├─ datainsights.png
│     ├─ datainsights1.png
│     ├─ predict1.png
│     └─ predict2.png
│     └─ predict3.png
└── scripts/ # Utility scripts

```
---

## Quickstart
```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
python -m src.models.train
streamlit run app/Home.py
```
---
## Screenshots

### Data Insights
![Data Insights Dashboard](reports/figures/datainsights.png)
![Data Insights Dashboard](reports/figures/datainsights2.png)

### Predict Churn
![Prediction Dashboard](reports/figures/predict1.png)
![Prediction Dashboard](reports/figures/predict2.png)
![Prediction Dashboard](reports/figures/predict3.png)

---

# Tech Stack

Python  
Pandas  
Scikit-Learn  
XGBoost  
Plotly  
Streamlit  

---

## Author

Olaoluwa Malachi
📧 olaoluwa.malachi@unb.ca

## License

This project is licensed under the MIT License.

