import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from src.models.predict import load_model, predict_proba
from src.utils.load_data import clean_telco

st.set_page_config(page_title="Predict — Telco Churn", layout="wide")
st.title("Predict Churn — ML Inference")

MODELS_DIR = Path("models")

model_choice = st.sidebar.selectbox("Model", ["logreg", "xgb"])
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

# Load model
try:
    model = load_model(model_choice)
except Exception as e:
    st.error(str(e))
    st.stop()

# Load meta to know expected feature columns
meta_path = MODELS_DIR / f"telco_churn_{model_choice}_meta.json"
expected_features = None
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    expected_features = meta.get("features", None)

st.subheader("Batch scoring (CSV)")
uploaded = st.file_uploader("Upload a CSV (same schema as training)", type=["csv"])

if uploaded is None:
    st.info("Tip: upload your Telco CSV to test quickly.")
    st.stop()

df_up = pd.read_csv(uploaded)
# Apply cleaning so numeric columns like TotalCharges don't break the model
from src.utils.load_data import clean_telco
df_up = clean_telco(df_up)

# Drop target if present
if "Churn" in df_up.columns:
    df_up = df_up.drop(columns=["Churn"])

# Keep customerID for output if present
customer_id = df_up["customerID"] if "customerID" in df_up.columns else None

# Remove customerID from features if present
X = df_up.drop(columns=["customerID"]) if "customerID" in df_up.columns else df_up

# Align columns to training expectation (handles missing/extra cols safely)
if expected_features is not None:
    # Add missing cols as NA
    missing = [c for c in expected_features if c not in X.columns]
    for c in missing:
        X[c] = pd.NA

    # Drop extra cols
    extra = [c for c in X.columns if c not in expected_features]
    if extra:
        X = X.drop(columns=extra)

    # Reorder
    X = X[expected_features]

    if missing:
        st.warning(f"Missing columns were added as empty: {missing}")
    if extra:
        st.warning(f"Extra columns were dropped: {extra}")

# Predict
try:
    probs = predict_proba(model, X)
except Exception as e:
    st.error("Prediction failed. Usually a schema mismatch or datatype issue.")
    st.code(str(e))
    st.stop()

preds = (probs >= threshold).astype(int)

out = pd.DataFrame({
    "churn_probability": probs.round(4),
    "predicted_churn": preds
})
if customer_id is not None:
    out.insert(0, "customerID", customer_id)

st.success("Predictions generated.")
st.write(out.head(20))

st.download_button(
    "Download predictions as CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="telco_churn_predictions.csv",
    mime="text/csv",
)