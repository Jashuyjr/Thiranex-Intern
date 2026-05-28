# ============================================================
# generate_dataset.py
# Task 2 - Predictive Modeling (Thiranex Internship)
# Creates a realistic Customer Churn dataset
# ============================================================
# WHAT IS CHURN?
#   A customer "churns" when they stop using a service.
#   We want to PREDICT which customers will churn BEFORE it happens.
# ============================================================

import pandas as pd
import numpy as np
import os

def create_churn_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic telecom customer churn dataset.

    Features:
      - tenure          : months the customer has been with company
      - monthly_charges : monthly bill amount (₹)
      - total_charges   : total amount paid so far (₹)
      - num_support_calls: times customer called support
      - contract_type   : Month-to-Month / One Year / Two Year
      - internet_service: DSL / Fiber optic / No
      - has_tech_support: Yes / No
      - age             : customer age
      - churn           : TARGET → 1 = churned, 0 = stayed
    """
    np.random.seed(seed)

    tenure           = np.random.randint(1, 72, n)
    monthly_charges  = np.round(np.random.uniform(300, 2000, n), 2)
    total_charges    = np.round(tenure * monthly_charges * np.random.uniform(0.9, 1.1, n), 2)
    num_support_calls= np.random.randint(0, 10, n)
    age              = np.random.randint(18, 70, n)

    contract_type    = np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n, p=[0.55, 0.25, 0.20]          # most customers are month-to-month
    )
    internet_service = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        n, p=[0.45, 0.35, 0.20]
    )
    has_tech_support = np.random.choice(["Yes", "No"], n, p=[0.40, 0.60])

    # ── Build churn probability (realistic business rules) ──
    # Higher churn if: short tenure, month-to-month, many support calls
    churn_prob = (
        0.40 * (tenure < 12).astype(float)          # new customers churn more
      + 0.25 * (contract_type == "Month-to-Month")
      + 0.20 * (num_support_calls > 5).astype(float)
      + 0.10 * (monthly_charges > 1500).astype(float)
      - 0.15 * (has_tech_support == "Yes")
      - 0.10 * (contract_type == "Two Year")
    )
    # Clip to valid probability range
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id"      : [f"CUST{str(i).zfill(4)}" for i in range(1, n+1)],
        "age"              : age,
        "tenure"           : tenure,
        "monthly_charges"  : monthly_charges,
        "total_charges"    : total_charges,
        "num_support_calls": num_support_calls,
        "contract_type"    : contract_type,
        "internet_service" : internet_service,
        "has_tech_support" : has_tech_support,
        "churn"            : churn,   # 1 = churned, 0 = stayed ← TARGET
    })

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/churn_dataset.csv", index=False)

    print("✅ Dataset created: 500 customers, 9 features + 1 target")
    print(f"   Churn rate: {churn.mean()*100:.1f}%  "
          f"({churn.sum()} churned / {n - churn.sum()} stayed)\n")
    return df


if __name__ == "__main__":
    df = create_churn_dataset()
    print(df.head(8).to_string(index=False))
