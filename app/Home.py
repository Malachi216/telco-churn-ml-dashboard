import streamlit as st

st.set_page_config(page_title="Telco Churn — Data + ML Dashboard", layout="wide")

st.title("Telco Customer Churn — Data Insights + ML Prediction")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose view", ["Data Insights", "Predict Churn"])

st.info("Project status: dashboard skeleton is live. Next: load dataset + charts + model pipeline.")

if mode == "Data Insights":
    st.subheader("Data Insights")
    st.write("✅ Add: KPI cards, churn breakdowns, interactive filters, segment insights.")
    st.write("Next files we’ll create: `src/utils/load_data.py`, `src/features/eda.py`.")

elif mode == "Predict Churn":
    st.subheader("Predict Churn")
    st.write("✅ Add: model selector, threshold slider, single-customer prediction, batch upload scoring.")
    st.write("Next files we’ll create: `src/models/train.py`, `src/models/predict.py` + saved model in `models/`.")