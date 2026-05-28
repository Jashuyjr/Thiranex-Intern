# ============================================================
# preprocessing.py
# Task 2 - Predictive Modeling (Thiranex Internship)
# Prepares raw data for ML models
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ──────────────────────────────────────────────
# STEP 1 : ENCODE CATEGORICAL COLUMNS
# ──────────────────────────────────────────────
# ML models only understand NUMBERS, not text like "Yes"/"No".
# Label Encoding converts each unique text value to a number.
# e.g.  "Month-to-Month" → 0,  "One Year" → 1,  "Two Year" → 2

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all object/string columns (except customer_id)
    into numeric codes using LabelEncoder.
    """
    print("=" * 55)
    print("  PREPROCESSING STEP 1 : ENCODE CATEGORIES")
    print("=" * 55)

    df = df.copy()
    le = LabelEncoder()

    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

    # Don't encode the ID column — it's not a feature
    cat_cols = [c for c in cat_cols if c != "customer_id"]

    for col in cat_cols:
        original_values = df[col].unique()
        df[col] = le.fit_transform(df[col])
        encoded_values = df[col].unique()
        print(f"   ✔ '{col}': {sorted(original_values)} → {sorted(encoded_values)}")

    print()
    return df


# ──────────────────────────────────────────────
# STEP 2 : SPLIT INTO FEATURES (X) AND TARGET (y)
# ──────────────────────────────────────────────
# X = all the input columns the model LEARNS FROM
# y = the column we want to PREDICT (churn: 0 or 1)

def split_features_target(df: pd.DataFrame,
                           target_col: str = "churn",
                           drop_cols: list = None):
    """
    Separates the dataset into:
      X → feature matrix (inputs)
      y → target vector  (what we predict)
    """
    print("=" * 55)
    print("  PREPROCESSING STEP 2 : SPLIT X AND y")
    print("=" * 55)

    drop = [target_col] + (drop_cols or [])
    X    = df.drop(columns=drop, errors="ignore")
    y    = df[target_col]

    print(f"   Features (X): {list(X.columns)}")
    print(f"   Target   (y): '{target_col}'  →  0=Stayed, 1=Churned")
    print(f"   X shape: {X.shape}   |   y shape: {y.shape}\n")
    return X, y


# ──────────────────────────────────────────────
# STEP 3 : TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
# We can't evaluate a model on the same data it trained on.
# So we reserve 20% of data as a "test set" it never sees.
#
#   80% → Training set  (model learns from this)
#   20% → Test set      (we evaluate on this)

def split_train_test(X, y, test_size: float = 0.2, seed: int = 42):
    """
    Splits data into training and testing subsets.
    stratify=y ensures both splits have the same churn ratio.
    """
    print("=" * 55)
    print("  PREPROCESSING STEP 3 : TRAIN / TEST SPLIT")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size  = test_size,
        random_state = seed,
        stratify   = y        # keeps churn ratio balanced
    )

    print(f"   Training samples : {len(X_train)}  ({(1-test_size)*100:.0f}%)")
    print(f"   Testing  samples : {len(X_test)}   ({test_size*100:.0f}%)")
    print(f"   Train churn rate : {y_train.mean()*100:.1f}%")
    print(f"   Test  churn rate : {y_test.mean()*100:.1f}%\n")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# STEP 4 : FEATURE SCALING
# ──────────────────────────────────────────────
# Logistic Regression is SENSITIVE to feature scales.
# tenure (1–72) vs total_charges (300–150000) — huge difference!
# StandardScaler makes every column have mean=0 and std=1.
# Decision Trees DON'T need scaling, but it doesn't hurt them.

def scale_features(X_train, X_test):
    """
    Fits StandardScaler on training data ONLY,
    then transforms both train and test sets.
    (Never fit on test data — that would be data leakage!)
    """
    print("=" * 55)
    print("  PREPROCESSING STEP 4 : FEATURE SCALING")
    print("=" * 55)

    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)         # transform only

    print("   ✔ StandardScaler applied (mean=0, std=1 per feature)")
    print("   ✔ Scaler fitted on TRAIN only (no data leakage)\n")

    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────
# FULL PIPELINE (convenience wrapper)
# ──────────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame):
    """
    Runs all 4 preprocessing steps in order.
    Returns everything needed for model training.
    """
    df_encoded                    = encode_categoricals(df)
    X, y                          = split_features_target(df_encoded, drop_cols=["customer_id"])
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist(), scaler


if __name__ == "__main__":
    # Quick test
    from generate_dataset import create_churn_dataset
    df = create_churn_dataset()
    X_train, X_test, y_train, y_test, feat_names, scaler = preprocess_pipeline(df)
    print(f"✅ Preprocessing complete. Ready for model training.")
    print(f"   X_train shape: {X_train.shape}")
