"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides inference for the Telco Churn model.
It is DECOUPLED from MLflow and loads flat artifacts copied into the container.

Artifacts expected inside Docker:
    /app/model/model.pkl
    /app/model/preprocessing.pkl
    /app/model/feature_columns.txt
"""

import os
import joblib
import pandas as pd

# ==========================================================
# MODEL ARTIFACT PATHS (Docker-safe, MLflow-free)
# ==========================================================
MODEL_PATH = "/app/model/model.pkl"
PREPROCESSING_PATH = "/app/model/preprocessing.pkl"
FEATURE_COLS_PATH = "/app/model/feature_columns.txt"

# ==========================================================
# Load model
# ==========================================================
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ==========================================================
# Load preprocessing metadata
# ==========================================================
try:
    preprocessing = joblib.load(PREPROCESSING_PATH)
    print(f"✅ Preprocessing metadata loaded from {PREPROCESSING_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load preprocessing metadata: {e}")

# ==========================================================
# Load feature columns (training-time order)
# ==========================================================
try:
    with open(FEATURE_COLS_PATH) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load feature columns: {e}")

# ==========================================================
# Feature transformation constants (MUST MATCH TRAINING)
# ==========================================================
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ==========================================================
# Feature transformation (serving-time)
# ==========================================================
def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as training.
    CRITICAL: Prevents train/serve skew.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # ---- Numeric coercion ----
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ---- Binary encoding ----
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # ---- One-hot encoding ----
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # ---- Boolean → int ----
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # ---- Align with training schema ----
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

# ==========================================================
# Public prediction function
# ==========================================================
def predict(input_dict: dict) -> str:
    """
    Generate churn prediction from raw customer input.
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])

    # Apply feature transformations
    df_enc = _serve_transform(df)

    # Run model inference
    try:
        pred = model.predict(df_enc)[0]
    except Exception as e:
        raise RuntimeError(f"❌ Model prediction failed: {e}")

    # Business-friendly output
    return "Likely to churn" if int(pred) == 1 else "Not likely to churn"
