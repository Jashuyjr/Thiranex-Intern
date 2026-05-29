# 🛒 Real-world Data Project — Retail E-Commerce
### Thiranex Internship — Task 4

---

## 🎯 Project Overview
End-to-end data science project on a **Retail E-Commerce** dataset.
Covers the full pipeline: data generation → cleaning → EDA →
customer segmentation → predictive modeling → findings.

**Business Question:** Which customers are likely to make a repeat
purchase, and what drives revenue & returns?

---

## 📁 Project Structure
```
task4-real-world-project/
├── main.py               ← 🚀 Run this (runs everything)
├── generate_dataset.py   ← Creates 1000-order e-commerce dataset
├── analysis.py           ← Cleaning, KPIs, EDA charts, RFM segments
├── prediction.py         ← ML model: predict repeat purchase
├── requirements.txt
├── README.md
└── output/
    ├── ecommerce_orders.csv      ← Raw dataset
    ├── rfm_segments.csv          ← Customer RFM table
    └── charts/
        ├── 01_monthly_revenue_trend.png
        ├── 02_category_analysis.png
        ├── 03_region_category_heatmap.png
        ├── 04_payment_shipping.png
        ├── 05_discount_impact.png
        ├── 06_customer_age_analysis.png
        ├── 07_rfm_segments.png
        ├── 08_prediction_confusion.png
        └── 09_prediction_roc.png
```

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
```

---

## 📊 Dataset — 19 Features, 1000 Orders

| Feature | Description |
|---|---|
| order_id | Unique order identifier |
| customer_id | Customer (400 unique → repeat buyers exist) |
| order_date | Date of purchase (Jan–Dec 2024) |
| category | Electronics / Clothing / Home & Kitchen / Books / Sports / Beauty |
| base_price | Original product price (₹) |
| discount_pct | Discount percentage applied |
| total_amount | Final amount paid (₹) |
| shipping_method | Standard / Express / Same-Day |
| shipping_days | Days taken to deliver |
| region | North / South / East / West / Central |
| customer_age | Age of customer |
| gender | Male / Female / Other |
| payment_method | UPI / Credit Card / Debit Card / Net Banking / COD |
| rating | Customer rating (1–5) |
| returned | Whether item was returned (1=Yes) |
| **repeat_purchase** | **Target: Will they buy again? (1=Yes)** |

---

## 🔬 Analysis Modules

### Module 2 — Analysis (analysis.py)
- **Data Cleaning:** Feature engineering, derived columns (net_revenue, profit, age_group)
- **Business KPIs:** Revenue, return rate, repeat rate, avg order value, top category/region
- **EDA (7 charts):** Revenue trends, category & regional analysis, discount impact, customer age groups
- **RFM Segmentation:** Champions, Loyal, Potential Loyalists, At Risk, Lost

### Module 3 — Prediction (prediction.py)
- **Models:** Logistic Regression + Decision Tree
- **Target:** repeat_purchase (0/1)
- **Validation:** 5-fold cross-validation + test set evaluation
- **Charts:** Confusion matrices + ROC curves

---

## 📌 Key Business Findings

| # | Finding |
|---|---|
| 1 | Electronics is the highest revenue category despite a 15% return rate |
| 2 | Month-to-Month shoppers need targeted retention offers |
| 3 | High discount (21–30%) drives repeat purchases but increases returns |
| 4 | UPI is the most popular payment method (35% of orders) |
| 5 | 25–34 age group spends most; 45–54 has the highest repeat rate |
| 6 | Customers with high ratings (≥4.0) are 2× more likely to repurchase |

---

## ✅ What to Submit
1. All 4 `.py` files
2. `output/ecommerce_orders.csv`
3. `output/rfm_segments.csv`
4. All 9 charts from `output/charts/`
5. This README
