from __future__ import annotations

from pathlib import Path
import pandas as pd


DEFAULT_PATH = Path("data/raw/telco_churn.csv")


def load_telco_csv(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """
    Loads and lightly cleans the Telco churn dataset.

    Expected columns (common Kaggle version):
    - customerID
    - Churn (Yes/No)
    - TotalCharges (string sometimes containing blanks)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. Put your CSV at data/raw/telco_churn.csv"
        )

    df = pd.read_csv(path)

    # Standardize column names (optional, keep original for clarity)
    # df.columns = [c.strip() for c in df.columns]

    # Clean TotalCharges: often has blanks which should become NaN then numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Ensure target exists
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip()
    else:
        raise ValueError("Expected a 'Churn' column (Yes/No). Not found in CSV.")

    # Drop exact duplicates (rare but safe)
    df = df.drop_duplicates()

    return df


def split_X_y(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Returns X (features), y (binary 0/1), and an id Series if available.
    """
    df = df.copy()

    # Keep customerID separately if present (helpful for UX)
    customer_id = df["customerID"] if "customerID" in df.columns else None

    y_raw = df[target_col].astype(str).str.lower()
    y = y_raw.map({"yes": 1, "no": 0})
    if y.isna().any():
        bad = df.loc[y.isna(), target_col].unique().tolist()
        raise ValueError(f"Unexpected Churn values found: {bad}")

    X = df.drop(columns=[target_col])

    return X, y.astype(int), customer_id

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning rules to any Telco dataframe (including uploaded CSVs).
    """
    df = df.copy()

    # Strip whitespace in object columns (prevents ' ' values)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # TotalCharges: blanks -> NaN, convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df