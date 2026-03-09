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
st.caption("Interactive churn analysis with business-focused segmentation and risk indicators.")

try:
    df = load_telco_csv()
except Exception as e:
    st.error(str(e))
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

contract_vals = ["All"]
if "Contract" in df.columns:
    contract_vals += sorted(df["Contract"].dropna().unique().tolist())
contract = st.sidebar.selectbox("Contract", contract_vals)

internet_vals = ["All"]
if "InternetService" in df.columns:
    internet_vals += sorted(df["InternetService"].dropna().unique().tolist())
internet = st.sidebar.selectbox("Internet Service", internet_vals)

# Create filtered dataframe
dff = df.copy()

if contract != "All" and "Contract" in dff.columns:
    dff = dff[dff["Contract"] == contract]

if internet != "All" and "InternetService" in dff.columns:
    dff = dff[dff["InternetService"] == internet]

# Working dataframe
dff2 = dff.copy()
dff2["ChurnBin"] = dff2["Churn"].str.lower().eq("yes").astype(int)

# -------------------------
# KPI row
# -------------------------
avg_tenure = dff2["tenure"].mean() if "tenure" in dff2.columns else None
avg_monthly = dff2["MonthlyCharges"].mean() if "MonthlyCharges" in dff2.columns else None

rev_risk = None
if "MonthlyCharges" in dff2.columns:
    rev_risk = dff2.loc[dff2["ChurnBin"] == 1, "MonthlyCharges"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(dff2):,}")
k2.metric("Churn rate", f"{dff2['ChurnBin'].mean()*100:.1f}%")
k3.metric("Avg tenure", f"{avg_tenure:.1f} mo" if avg_tenure is not None else "—")
k4.metric("Revenue at risk / month", f"${rev_risk:,.0f}" if rev_risk is not None else "—")

if avg_monthly is not None:
    st.caption(f"Average monthly charge: ${avg_monthly:.2f}")

st.divider()

# -------------------------
# Row 1
# -------------------------
left, right = st.columns(2)

with left:
    st.subheader("Churn Distribution")
    churn_counts = dff2["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    fig = px.bar(churn_counts, x="Churn", y="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn by Contract")
    if "Contract" in dff2.columns:
        tmp = (
            dff2.groupby("Contract")["ChurnBin"]
            .mean()
            .reset_index()
            .rename(columns={"ChurnBin": "ChurnRate"})
        )
        fig = px.bar(tmp, x="Contract", y="ChurnRate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contract column not found in your dataset.")

# -------------------------
# Row 2
# -------------------------
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

# -------------------------
# Row 3
# -------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Monthly Charges by Churn")
    if "MonthlyCharges" in dff2.columns:
        fig = px.box(dff2, x="Churn", y="MonthlyCharges")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("MonthlyCharges column not found.")

with c2:
    st.subheader("Tenure by Churn")
    if "tenure" in dff2.columns:
        fig = px.box(dff2, x="Churn", y="tenure")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("tenure column not found in your dataset.")

# -------------------------
# Row 4
# -------------------------
if "tenure" in dff2.columns:
    st.subheader("Tenure Band vs Churn")
    dff2["tenure_band"] = pd.cut(
        dff2["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True
    )
    tmp = (
        dff2.groupby("tenure_band", observed=False)["ChurnBin"]
        .mean()
        .reset_index()
        .rename(columns={"ChurnBin": "ChurnRate"})
    )
    fig = px.bar(tmp, x="tenure_band", y="ChurnRate")
    st.plotly_chart(fig, use_container_width=True)


    #done marker final