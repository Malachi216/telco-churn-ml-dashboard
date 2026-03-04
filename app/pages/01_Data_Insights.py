import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.load_data import load_telco_csv

st.set_page_config(page_title="Data Insights — Telco Churn", layout="wide")
st.title("Data Insights — Telco Customer Churn")

try:
    df = load_telco_csv()
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Filters (keep minimal and useful)
st.sidebar.header("Filters")

contract_vals = ["All"]
if "Contract" in df.columns:
    contract_vals += sorted(df["Contract"].dropna().unique().tolist())
contract = st.sidebar.selectbox("Contract", contract_vals)

internet_vals = ["All"]
if "InternetService" in df.columns:
    internet_vals += sorted(df["InternetService"].dropna().unique().tolist())
internet = st.sidebar.selectbox("Internet Service", internet_vals)

dff = df.copy()
if contract != "All" and "Contract" in dff.columns:
    dff = dff[dff["Contract"] == contract]
if internet != "All" and "InternetService" in dff.columns:
    dff = dff[dff["InternetService"] == internet]

# --- KPIs
col1, col2, col3 = st.columns(3)

churn_rate = (dff["Churn"].str.lower() == "yes").mean() if "Churn" in dff.columns else 0.0
col1.metric("Rows (filtered)", f"{len(dff):,}")
col2.metric("Churn rate", f"{churn_rate*100:.1f}%")

if "MonthlyCharges" in dff.columns:
    col3.metric("Avg MonthlyCharges", f"{dff['MonthlyCharges'].mean():.2f}")
else:
    col3.metric("Avg MonthlyCharges", "—")

st.divider()

# --- Charts
left, right = st.columns(2)

with left:
    st.subheader("Churn Distribution")
    churn_counts = dff["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    fig = px.bar(churn_counts, x="Churn", y="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn by Contract")
    if "Contract" in dff.columns:
        tmp = (
            dff.assign(churn_bin=dff["Churn"].str.lower().eq("yes").astype(int))
            .groupby("Contract")["churn_bin"]
            .mean()
            .reset_index()
            .rename(columns={"churn_bin": "ChurnRate"})
        )
        fig = px.bar(tmp, x="Contract", y="ChurnRate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contract column not found in your dataset.")

st.subheader("Tenure vs Churn")
if "tenure" in dff.columns:
    dff2 = dff.copy()
    dff2["ChurnBin"] = dff2["Churn"].str.lower().eq("yes").astype(int)
    fig = px.box(dff2, x="Churn", y="tenure")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("tenure column not found in your dataset.")

st.caption("Next: add deeper segments (high-value at risk), and a ‘key drivers’ section.")
