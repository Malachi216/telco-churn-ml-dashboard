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

# col1, col2, col3 = st.columns(3)

# churn_rate = (dff["Churn"].str.lower() == "yes").mean() if "Churn" in dff.columns else 0.0
# col1.metric("Rows (filtered)", f"{len(dff):,}")
# col2.metric("Churn rate", f"{churn_rate*100:.1f}%")

# if "MonthlyCharges" in dff.columns:
#     col3.metric("Avg MonthlyCharges", f"{dff['MonthlyCharges'].mean():.2f}")
# else:
#     col3.metric("Avg MonthlyCharges", "—")

# st.divider()
dff2 = dff.copy()
dff2["ChurnBin"] = dff2["Churn"].str.lower().eq("yes").astype(int)

avg_tenure = dff2["tenure"].mean() if "tenure" in dff2.columns else None
avg_monthly = dff2["MonthlyCharges"].mean() if "MonthlyCharges" in dff2.columns else None
churn_rate = dff2["ChurnBin"].mean()

rev_risk = None
if "MonthlyCharges" in dff2.columns:
    rev_risk = dff2.loc[dff2["ChurnBin"] == 1, "MonthlyCharges"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(dff2):,}")
k2.metric("Churn rate", f"{churn_rate*100:.1f}%")
k3.metric("Avg tenure", f"{avg_tenure:.1f} mo" if avg_tenure is not None else "—")
k4.metric("Monthly revenue at risk", f"${rev_risk:,.0f}" if rev_risk is not None else "—")

st.caption(f"Average monthly charge: ${avg_monthly:.2f}" if avg_monthly is not None else "")
st.divider()

l1, l2 = st.columns(2)

with l1:
    st.subheader("Churn by Internet Service")
    if "InternetService" in dff2.columns:
        tmp = (
            dff2.groupby("InternetService")["ChurnBin"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnBin": "ChurnRate"})
        )
        fig = px.bar(tmp, x="InternetService", y="ChurnRate")
        st.plotly_chart(fig, use_container_width=True)

with l2:
    st.subheader("Churn by Payment Method")
    if "PaymentMethod" in dff2.columns:
        tmp = (
            dff2.groupby("PaymentMethod")["ChurnBin"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnBin": "ChurnRate"})
        )
        fig = px.bar(tmp, x="PaymentMethod", y="ChurnRate")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly Charges by Churn")
if "MonthlyCharges" in dff2.columns:
    fig = px.box(dff2, x="Churn", y="MonthlyCharges")
    st.plotly_chart(fig, use_container_width=True)

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

if "tenure" in dff2.columns:
    st.subheader("Tenure Band vs Churn")
    dff2["tenure_band"] = pd.cut(
        dff2["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True
    )
    tmp = (
        dff2.groupby("tenure_band")["ChurnBin"]
        .mean()
        .reset_index()
        .rename(columns={"ChurnBin": "ChurnRate"})
    )
    fig = px.bar(tmp, x="tenure_band", y="ChurnRate")
    st.plotly_chart(fig, use_container_width=True)
