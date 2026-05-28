# ============================================================
# data_cleaning.py
# Task 1 - Data Cleaning & Visualization (Thiranex Internship)
# Author  : [Your Name]
# Date    : May 2026
# ============================================================
# HOW TO RUN:
#   python data_cleaning.py
# ============================================================

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# SECTION 1 : LOAD DATA
# ──────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    Replace 'sample_data.csv' with your own file path.
    """
    df = pd.read_csv(filepath)
    print("✅ Data loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


def create_sample_data() -> pd.DataFrame:
    """
    Creates a realistic sample sales dataset so you can run the
    project right away — swap in your real CSV when ready.
    """
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "order_id"   : range(1, n + 1),
        "customer_age": np.random.randint(18, 65, n).astype(float),
        "product"    : np.random.choice(["Laptop", "Phone", "Tablet", "Watch"], n),
        "category"   : np.random.choice(["Electronics", "Accessories"], n),
        "quantity"   : np.random.randint(1, 10, n).astype(float),
        "unit_price" : np.round(np.random.uniform(50, 2000, n), 2),
        "region"     : np.random.choice(["North", "South", "East", "West"], n),
        "rating"     : np.random.uniform(1, 5, n).round(1),
        "sale_date"  : pd.date_range("2024-01-01", periods=n, freq="2D"),
    })

    # --- Inject realistic "dirty data" problems ---

    # 1. Missing values (~8 % of cells)
    for col in ["customer_age", "quantity", "rating"]:
        missing_idx = np.random.choice(df.index, size=int(n * 0.08), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2. Duplicate rows (5 exact copies)
    duplicates = df.sample(5, random_state=1)
    df = pd.concat([df, duplicates], ignore_index=True)

    # 3. Outliers in unit_price (a few extreme values)
    df.loc[0, "unit_price"]   = 50000   # way too high
    df.loc[1, "unit_price"]   = -200    # negative — impossible
    df.loc[2, "customer_age"] = 150     # impossible age

    print("✅ Sample dataset created with intentional dirty data.\n")
    return df


# ──────────────────────────────────────────────
# SECTION 2 : INSPECT DATA
# ──────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("=" * 55)
    print("  STEP 1 : BASIC INSPECTION")
    print("=" * 55)

    print("\n📋 First 5 rows:")
    print(df.head())

    print("\n📊 Dataset info (dtypes + non-null counts):")
    df.info()

    print("\n📈 Statistical summary (numerical columns):")
    print(df.describe().round(2))


# ──────────────────────────────────────────────
# SECTION 3 : HANDLE MISSING VALUES
# ──────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      • Numerical columns  → fill with MEDIAN  (robust to outliers)
      • Categorical columns → fill with MODE   (most frequent value)
    """
    print("\n" + "=" * 55)
    print("  STEP 2 : HANDLING MISSING VALUES")
    print("=" * 55)

    # Count missing before
    missing_before = df.isnull().sum()
    missing_pct    = (missing_before / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count" : missing_before,
        "Missing %"     : missing_pct
    })
    missing_report = missing_report[missing_report["Missing Count"] > 0]

    print("\n🔍 Missing values BEFORE cleaning:")
    print(missing_report.to_string())

    # --- Fill numerical columns with median ---
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            print(f"   ✔ '{col}' → filled {missing_before[col]} NaNs with median ({median_val:.2f})")

    # --- Fill categorical columns with mode ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            print(f"   ✔ '{col}' → filled NaNs with mode ('{mode_val}')")

    # Verify
    remaining = df.isnull().sum().sum()
    print(f"\n✅ Missing values AFTER cleaning: {remaining} remaining")
    return df


# ──────────────────────────────────────────────
# SECTION 4 : REMOVE DUPLICATES
# ──────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows.
    Keeps the first occurrence and drops the rest.
    """
    print("\n" + "=" * 55)
    print("  STEP 3 : HANDLING DUPLICATES")
    print("=" * 55)

    before = len(df)
    df     = df.drop_duplicates()
    after  = len(df)
    removed = before - after

    print(f"\n🔍 Rows before: {before}")
    print(f"   Duplicates removed: {removed}")
    print(f"✅ Rows after: {after}")
    return df


# ──────────────────────────────────────────────
# SECTION 5 : HANDLE OUTLIERS
# ──────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Uses the IQR (Inter-Quartile Range) method to detect outliers.

    Any value below  Q1 - 1.5×IQR  or above  Q3 + 1.5×IQR
    is capped (clipped) to the boundary instead of being deleted.
    Capping is safer than deletion for small datasets.
    """
    print("\n" + "=" * 55)
    print("  STEP 4 : HANDLING OUTLIERS (IQR Method)")
    print("=" * 55)

    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count outliers before capping
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        # Cap (clip) the outliers
        df[col] = df[col].clip(lower=lower, upper=upper)

        print(f"\n📌 Column: '{col}'")
        print(f"   IQR range: [{lower:.2f}, {upper:.2f}]")
        print(f"   Outliers capped: {outlier_count}")

    print("\n✅ Outlier handling complete.")
    return df


# ──────────────────────────────────────────────
# SECTION 6 : FEATURE ENGINEERING (BONUS)
# ──────────────────────────────────────────────

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived columns for richer visualizations later.
    """
    print("\n" + "=" * 55)
    print("  STEP 5 : FEATURE ENGINEERING")
    print("=" * 55)

    # Total revenue per order
    df["total_revenue"] = (df["quantity"] * df["unit_price"]).round(2)

    # Extract month name from sale_date (if column exists)
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df["month"]     = df["sale_date"].dt.strftime("%b")  # e.g., Jan, Feb

    # Age group bucket
    if "customer_age" in df.columns:
        bins   = [0,  25,  35,  45,  65, 200]
        labels = ["<25", "25-34", "35-44", "45-64", "65+"]
        df["age_group"] = pd.cut(df["customer_age"], bins=bins, labels=labels)

    print("   ✔ Added: total_revenue, month, age_group")
    print(f"✅ Final dataset shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
# SECTION 7 : SAVE CLEANED DATA
# ──────────────────────────────────────────────

def save_clean_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Cleaned data saved to: {output_path}")


# ──────────────────────────────────────────────
# MAIN — RUN ALL STEPS IN ORDER
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # STEP 0: Create sample data (replace with load_data("your_file.csv"))
    df = create_sample_data()

    # Uncomment the line below to load YOUR real dataset instead:
    # df = load_data("data/your_dataset.csv")

    # STEP 1: Inspect
    inspect_data(df)

    # STEP 2: Fix missing values
    df = handle_missing_values(df)

    # STEP 3: Remove duplicates
    df = remove_duplicates(df)

    # STEP 4: Fix outliers (list the numerical columns you care about)
    df = handle_outliers(df, columns=["unit_price", "customer_age", "rating"])

    # STEP 5: Add derived columns
    df = add_derived_columns(df)

    # STEP 6: Save
    save_clean_data(df, "output/cleaned_data.csv")

    print("\n🎉 Data Cleaning complete! Run 'python visualizations.py' next.")
