# ============================================================
# generate_dataset.py
# Task 4 - Real-world Data Project : Retail E-Commerce
# Thiranex Internship
# ============================================================

import pandas as pd
import numpy as np
import os

def create_ecommerce_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    order_ids    = [f"ORD{str(i).zfill(5)}" for i in range(1, n + 1)]
    customer_ids = [f"CUST{str(np.random.randint(1, 401)).zfill(4)}"
                    for _ in range(n)]

    order_dates  = pd.date_range("2024-01-01", "2024-12-31", periods=n)
    order_dates  = order_dates[np.random.permutation(n)]

    categories   = np.random.choice(
        ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty"],
        n, p=[0.25, 0.20, 0.20, 0.10, 0.15, 0.10]
    )

    base_price = np.where(categories == "Electronics",  np.random.uniform(500,  8000, n),
                 np.where(categories == "Clothing",      np.random.uniform(200,  2000, n),
                 np.where(categories == "Home & Kitchen",np.random.uniform(300,  5000, n),
                 np.where(categories == "Books",         np.random.uniform(100,   800, n),
                 np.where(categories == "Sports",        np.random.uniform(150,  3000, n),
                                                         np.random.uniform(100,  1500, n))))))

    quantity      = np.random.randint(1, 6, n)
    discount_pct  = np.random.choice([0,5,10,15,20,25,30], n,
                                      p=[0.30,0.15,0.20,0.15,0.10,0.05,0.05])
    discount_amt  = np.round(base_price * discount_pct / 100, 2)
    final_price   = np.round(base_price - discount_amt, 2)
    total_amount  = np.round(final_price * quantity, 2)

    shipping_methods = np.random.choice(
        ["Standard","Express","Same-Day"], n, p=[0.55, 0.30, 0.15])
    shipping_days = np.where(shipping_methods == "Standard",
                             np.random.randint(4,8,n),
                    np.where(shipping_methods == "Express",
                             np.random.randint(2,4,n),
                             np.random.randint(1,2,n)))

    regions      = np.random.choice(
        ["North","South","East","West","Central"], n, p=[0.20,0.25,0.20,0.20,0.15])
    customer_age = np.random.randint(18, 65, n)
    gender       = np.random.choice(["Male","Female","Other"], n, p=[0.48,0.48,0.04])

    return_prob  = np.where(categories=="Electronics", 0.15,
                   np.where(categories=="Clothing",    0.20,
                   np.where(categories=="Beauty",      0.12, 0.08)))
    returned     = (np.random.rand(n) < return_prob).astype(int)

    payment      = np.random.choice(
        ["UPI","Credit Card","Debit Card","Net Banking","COD"],
        n, p=[0.35,0.25,0.20,0.10,0.10])

    ratings      = np.round(np.random.uniform(2.5, 5.0, n), 1)
    ratings      = np.where(returned==1, np.clip(ratings-1.0, 1.0, 5.0), ratings)

    repeat_prob  = (
        0.30 * (ratings >= 4.0).astype(float)
      + 0.20 * (returned == 0).astype(float)
      + 0.15 * (discount_pct >= 10).astype(float)
      + 0.10 * (shipping_methods != "Standard").astype(float)
      + 0.15 * np.isin(categories, ["Electronics","Sports"]).astype(float)
      + 0.05 * (total_amount > 2000).astype(float)
    )
    repeat_prob      = np.clip(repeat_prob, 0.05, 0.95)
    repeat_purchase  = (np.random.rand(n) < repeat_prob).astype(int)

    df = pd.DataFrame({
        "order_id"        : order_ids,
        "customer_id"     : customer_ids,
        "order_date"      : order_dates,
        "category"        : categories,
        "base_price"      : np.round(base_price, 2),
        "discount_pct"    : discount_pct,
        "discount_amt"    : discount_amt,
        "final_price"     : final_price,
        "quantity"        : quantity,
        "total_amount"    : total_amount,
        "shipping_method" : shipping_methods,
        "shipping_days"   : shipping_days,
        "region"          : regions,
        "customer_age"    : customer_age,
        "gender"          : gender,
        "payment_method"  : payment,
        "rating"          : ratings,
        "returned"        : returned,
        "repeat_purchase" : repeat_purchase,
    })

    df = df.sort_values("order_date").reset_index(drop=True)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/ecommerce_orders.csv", index=False)

    print("✅ E-Commerce dataset created!")
    print(f"   Orders          : {len(df)}")
    print(f"   Unique customers: {df['customer_id'].nunique()}")
    print(f"   Date range      : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
    print(f"   Return rate     : {df['returned'].mean()*100:.1f}%")
    print(f"   Repeat buyers   : {df['repeat_purchase'].mean()*100:.1f}%\n")
    return df


if __name__ == "__main__":
    df = create_ecommerce_dataset()
    print(df.head(5).to_string(index=False))
