# 🔍 Exploratory Data Analysis (EDA) Project
### Thiranex Internship — Task 3

---

## 🎯 Objective
Analyze the Customer Churn dataset to uncover patterns, trends, and key
factors that influence whether a customer leaves or stays.

---

## 📁 Project Structure
```
task3-eda/
├── eda_analysis.py     ← 🚀 Run this (everything in one file)
├── requirements.txt
├── README.md
└── output/
    ├── churn_dataset.csv
    └── charts/
        ├── 01_univariate_analysis.png
        ├── 02_bivariate_numerical_vs_churn.png
        ├── 03_bivariate_categorical_vs_churn.png
        ├── 04_correlation_heatmap.png
        ├── 05_scatter_tenure_charges.png
        ├── 06_churn_risk_heatmap.png
        └── 07_violin_tenure_by_contract.png
```

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python eda_analysis.py
```

---

## 📊 EDA Approach — 3 Levels of Analysis

### Level 1 — Univariate (1 variable at a time)
- Histograms + KDE for numerical columns
- Count charts for categorical columns
- Churn split pie chart

### Level 2 — Bivariate (feature vs churn)
- Box plots: numerical features vs churn label
- Bar charts: churn rate per category group

### Level 3 — Multivariate (3+ variables together)
- Correlation heatmap (lower triangle)
- Scatter + KDE density: tenure vs charges coloured by churn
- Risk heatmap: contract type × support call volume
- Violin plot: tenure by contract type split by churn

---

## 📌 Key Findings

| # | Insight |
|---|---------|
| 1 | Overall churn rate is ~29% — high enough to need action |
| 2 | Churned customers have significantly shorter tenure (newer = riskier) |
| 3 | Month-to-Month contracts have the highest churn rate |
| 4 | Customers with >5 support calls churn much more than low-call customers |
| 5 | Having tech support reduces churn noticeably |
| 6 | `num_support_calls` and `tenure` are the strongest churn predictors |

---

## ✅ What to Submit
1. `eda_analysis.py`
2. All 7 chart PNGs from `output/charts/`
3. This README
