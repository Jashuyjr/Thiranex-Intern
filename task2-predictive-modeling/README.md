# 🤖 Predictive Modeling Using Machine Learning
### Thiranex Internship — Task 2

---

## 🎯 Problem Statement
**Customer Churn Prediction** — Predict whether a telecom customer will
leave (churn = 1) or stay (churn = 0) based on their usage patterns.

**Why it matters:** Acquiring a new customer costs 5–7× more than retaining one.
Predicting churn early lets businesses take action before it's too late.

---

## 📁 Project Structure
```
task2_predictive_modeling/
├── main.py               ← 🚀 RUN THIS FILE (runs everything)
├── generate_dataset.py   ← Creates sample churn dataset
├── preprocessing.py      ← Encode, split, scale data
├── models.py             ← Train, evaluate, visualize models
├── requirements.txt      ← Libraries needed
└── output/
    ├── churn_dataset.csv ← Generated dataset
    └── charts/
        ├── 01_confusion_matrices.png
        ├── 02_roc_curves.png
        ├── 03_feature_importance.png
        ├── 04_decision_tree_structure.png
        └── 05_model_comparison.png
```

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run (Just One Command!)
```bash
python main.py
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Customer age |
| tenure | Numerical | Months with company |
| monthly_charges | Numerical | Monthly bill (₹) |
| total_charges | Numerical | Total paid so far (₹) |
| num_support_calls | Numerical | Times called support |
| contract_type | Categorical | Month-to-Month / 1yr / 2yr |
| internet_service | Categorical | DSL / Fiber / No |
| has_tech_support | Categorical | Yes / No |
| **churn** | **Target** | **1=Churned, 0=Stayed** |

---

## 🧠 Models Used

### 1. Logistic Regression
- The classic "linear" classifier
- Learns a straight-line decision boundary
- Fast, interpretable, good baseline
- Needs feature scaling (StandardScaler applied)

### 2. Decision Tree Classifier
- Learns a flowchart of YES/NO questions
- Naturally handles non-linear patterns
- Easy to visualize and explain
- max_depth=5 to prevent overfitting

---

## 📈 Key Metrics Explained

| Metric | What it means | Why it matters |
|---|---|---|
| Accuracy | % of correct predictions | General performance |
| Precision | Of predicted churners, how many actually churned | Avoid false alarms |
| Recall | Of actual churners, how many did we catch | Most critical for churn! |
| F1-Score | Balance of Precision & Recall | Best overall metric |
| AUC-ROC | Area under ROC curve (1.0 = perfect) | Threshold-independent |

---

## 📊 Visualizations

| Chart | What it shows |
|---|---|
| Confusion Matrix | TP, TN, FP, FN counts for both models |
| ROC Curves | True Positive Rate vs False Positive Rate |
| Feature Importance | Which features matter most (Decision Tree) |
| Tree Structure | The actual flowchart of decisions |
| Model Comparison | Side-by-side metric comparison |

---

## ✅ What to Submit
1. All 4 `.py` files
2. `output/churn_dataset.csv`
3. All 5 charts from `output/charts/`
4. This README
