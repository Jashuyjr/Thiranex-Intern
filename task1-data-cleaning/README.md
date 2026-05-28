# 📊 Data Cleaning & Visualization Project
### Thiranex Internship — Task 1

---

## 📁 Project Structure

```
data_science_project/
├── data_cleaning.py       ← STEP 1: Run this first
├── visualizations.py      ← STEP 2: Run this after cleaning
├── requirements.txt       ← Libraries to install
└── output/
    ├── cleaned_data.csv   ← Auto-created after step 1
    └── charts/            ← All chart PNGs saved here
        ├── 01_revenue_by_product.png
        ├── 02_correlation_heatmap.png
        ├── 03_price_by_region.png
        └── 04_monthly_revenue_trend.png
```

---

## ⚙️ Setup (Do This Once)

Open your VS Code terminal and run:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Data Cleaning
```bash
python data_cleaning.py
```
This will:
- Create a sample dirty dataset (or load your own CSV)
- Fix missing values, remove duplicates, cap outliers
- Add derived columns (total_revenue, month, age_group)
- Save `output/cleaned_data.csv`

### Step 2 — Visualizations
```bash
python visualizations.py
```
This will:
- Load the cleaned CSV
- Generate 4 professional charts
- Save all charts as PNGs in `output/charts/`

---

## 🔄 Using Your OWN Dataset

In `data_cleaning.py`, replace this line:
```python
df = create_sample_data()
```
With:
```python
df = load_data("data/your_filename.csv")
```

Then update `handle_outliers(df, columns=[...])` with your actual column names.

---

## 📊 Visualizations Explained

| # | Chart Type | What It Shows | Why It Matters |
|---|-----------|---------------|----------------|
| 1 | Bar Chart | Revenue per product | Which product earns most |
| 2 | Heatmap | Correlation matrix | Relationships between variables |
| 3 | Box + Strip | Price spread per region | Outliers & regional differences |
| 4 | Line Chart | Monthly revenue trend | Business growth over time |

---

## 🐛 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'seaborn'` | Run `pip install seaborn` |
| `FileNotFoundError: cleaned_data.csv` | Run `data_cleaning.py` first |
| Charts not showing on screen | You can still open the saved PNGs in `output/charts/` |

---

## ✅ What to Submit

1. Both `.py` files
2. `output/cleaned_data.csv`
3. All 4 chart PNGs from `output/charts/`
4. A short summary (3–5 sentences) of what you found in the data
