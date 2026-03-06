import streamlit as st

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("📉 Telco Customer Churn — Portfolio Dashboard")
st.caption("Interactive analytics + deployable churn prediction system (LogReg / XGBoost).")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("🔎 Data Insights")
    st.write("- KPI cards + filters\n- Churn segmentation\n- Revenue-at-risk view")
    st.info("Open from the left sidebar: **Data Insights**")

with c2:
    st.subheader("🤖 Predict Churn")
    st.write("- Model metrics (ROC/PR)\n- Threshold slider\n- Batch scoring + download")
    st.info("Open from the left sidebar: **Predict Churn**")

with c3:
    st.subheader("🧾 What this proves")
    st.write("- Data cleaning pipeline\n- Model training + persistence\n- Deployable app UX\n- Business framing")
    st.success("Recruiter-ready end-to-end project")

st.divider()

st.markdown(
    """
### Quickstart
1. Train models: `python -m src.models.train`  
2. Run app: `streamlit run app/Home.py`

### Notes
- Dataset not committed (kept in `data/raw/`)
- Models saved in `models/`
"""
)

#done marker