import sys
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import plotly.express as px
from src.models.explain import get_xgb_feature_importance, get_single_row_contributions_logreg
from src.utils.load_data import clean_telco

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from src.models.predict import load_model, predict_proba
from src.utils.load_data import clean_telco

st.set_page_config(page_title="Predict — Telco Churn", layout="wide")
st.title("Predict Churn — ML Inference")
st.caption("Batch scoring, threshold tuning, model evaluation, and explainability.")

st.divider()
st.subheader("Single-customer explanation")

try:
    source_df = df_up.copy()

    if len(source_df) > 0:
        row_index = st.number_input(
            "Select row index from uploaded file",
            min_value=0,
            max_value=len(source_df) - 1,
            value=0,
            step=1
        )

        row_raw = source_df.iloc[[row_index]].copy()

        if "Churn" in row_raw.columns:
            row_raw = row_raw.drop(columns=["Churn"])
        row_customer_id = row_raw["customerID"].iloc[0] if "customerID" in row_raw.columns else None
        row_X = row_raw.drop(columns=["customerID"]) if "customerID" in row_raw.columns else row_raw
        row_X = clean_telco(row_X)

        if expected_features is not None:
            for c in expected_features:
                if c not in row_X.columns:
                    row_X[c] = pd.NA
            extra = [c for c in row_X.columns if c not in expected_features]
            if extra:
                row_X = row_X.drop(columns=extra)
            row_X = row_X[expected_features]

        row_prob = float(predict_proba(model, row_X).iloc[0])
        row_pred = int(row_prob >= threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Customer ID", str(row_customer_id) if row_customer_id is not None else f"Row {row_index}")
        c2.metric("Churn probability", f"{row_prob:.3f}")
        c3.metric("Predicted churn", "Yes" if row_pred == 1 else "No")

        if model_choice == "logreg":
            local_df = get_single_row_contributions_logreg(model, row_X, row_index=0, top_n=12)
            local_df["direction"] = local_df["contribution"].apply(lambda x: "Increase risk" if x > 0 else "Reduce risk")

            fig_local = px.bar(
                local_df.sort_values("contribution"),
                x="contribution",
                y="feature",
                color="direction",
                orientation="h",
                title="Top local drivers for this prediction (Logistic Regression)"
            )
            st.plotly_chart(fig_local, use_container_width=True)
        else:
            st.info("Local per-customer contribution chart is enabled for Logistic Regression in this version. Use LogReg to view it.")

except Exception as e:
    st.warning("Could not generate single-customer explanation.")
    st.code(str(e))

MODELS_DIR = Path("models")

model_choice = st.sidebar.selectbox("Model", ["logreg", "xgb"])
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)

eval_dir = Path("models/eval")
if eval_dir.exists():
    try:
        X_test = pd.read_parquet(eval_dir / "X_test.parquet")
        y_test = pd.read_parquet(eval_dir / "y_test.parquet")["y"]

        # Align test features same way
        if expected_features is not None:
            for c in expected_features:
                if c not in X_test.columns:
                    X_test[c] = pd.NA
            extra = [c for c in X_test.columns if c not in expected_features]
            if extra:
                X_test = X_test.drop(columns=extra)
            X_test = X_test[expected_features]

        p = predict_proba(model, X_test)
        y_pred = (p >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.subheader("Model performance at selected threshold")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1", f"{f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix (TN FP / FN TP)")
        st.code(cm)

    except Exception as e:
        st.warning("Evaluation set exists but could not be loaded.")
        st.code(str(e))

if meta_path.exists():
    st.subheader("Saved model summary")
    a, b, c, d = st.columns(4)
    a.metric("Model", meta.get("model_name", model_choice).upper())
    b.metric("ROC-AUC", f"{meta.get('roc_auc', 0):.3f}")
    c.metric("PR-AUC", f"{meta.get('pr_auc', 0):.3f}")
    d.metric("Test rows", f"{meta.get('n_test', 0):,}")

st.divider()
st.subheader("Global feature importance")

try:
    eval_dir = Path("models/eval")
    X_eval = pd.read_parquet(eval_dir / "X_test.parquet")

    if expected_features is not None:
        for c in expected_features:
            if c not in X_eval.columns:
                X_eval[c] = pd.NA
        extra = [c for c in X_eval.columns if c not in expected_features]
        if extra:
            X_eval = X_eval.drop(columns=extra)
        X_eval = X_eval[expected_features]

    if model_choice == "xgb":
        imp_df = get_xgb_feature_importance(model, X_eval, top_n=15)
        fig_imp = px.bar(
            imp_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 XGBoost Feature Importances"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Global feature importance chart is shown for XGBoost. Switch model to XGB to view it.")

except Exception as e:
    st.warning("Could not generate global feature importance.")
    st.code(str(e))

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

if expected_features is not None and meta_path.exists():
    st.sidebar.markdown("### Model metrics")
    st.sidebar.write(f"ROC-AUC: **{meta['roc_auc']:.3f}**")
    st.sidebar.write(f"PR-AUC: **{meta['pr_auc']:.3f}**")

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