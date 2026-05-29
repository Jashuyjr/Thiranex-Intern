# ============================================================
# main.py
# Task 4 - Real-world Data Project : Retail E-Commerce
# Thiranex Internship
# ============================================================
# 🚀 SINGLE COMMAND TO RUN EVERYTHING:
#    python main.py
#
# PIPELINE:
#   1. generate_dataset.py → 1000 realistic e-commerce orders
#   2. analysis.py         → clean, KPIs, 7 EDA charts, RFM
#   3. prediction.py       → ML model, 2 evaluation charts
#   4. Final summary printed
# ============================================================

from generate_dataset import create_ecommerce_dataset
from analysis         import (clean_and_engineer, print_business_kpis,
                               plot_monthly_revenue, plot_category_analysis,
                               plot_regional_revenue, plot_payment_and_shipping,
                               plot_discount_impact, plot_customer_age_analysis,
                               rfm_segmentation, plot_rfm_segments)
from prediction       import prepare_features, train_models, evaluate_and_plot

import pandas as pd


def banner(text):
    print("\n╔" + "═"*56 + "╗")
    print(f"║  {text:<54}║")
    print("╚" + "═"*56 + "╝\n")


if __name__ == "__main__":

    banner("TASK 4 — Real-world Data Project : Retail E-Commerce")

    # ── MODULE 1: Dataset ────────────────────────────────────
    banner("MODULE 1 : Generate E-Commerce Dataset")
    df_raw = create_ecommerce_dataset(n=1000)

    # ── MODULE 2: Analysis ───────────────────────────────────
    banner("MODULE 2 : Data Cleaning + EDA + RFM")
    df  = clean_and_engineer(df_raw)
    print_business_kpis(df)

    print("  ── EDA Visualizations ──")
    plot_monthly_revenue(df)
    plot_category_analysis(df)
    plot_regional_revenue(df)
    plot_payment_and_shipping(df)
    plot_discount_impact(df)
    plot_customer_age_analysis(df)

    rfm = rfm_segmentation(df)
    plot_rfm_segments(rfm)

    # ── MODULE 3: Prediction ─────────────────────────────────
    banner("MODULE 3 : Predict Repeat Purchase (ML)")
    X_tr, X_te, y_tr, y_te, feat_names = prepare_features(df)
    models  = train_models(X_tr, y_tr)
    metrics = evaluate_and_plot(models, X_te, y_te, feat_names)

    # ── Final Summary ────────────────────────────────────────
    banner("FINAL SUMMARY")
    print("📊 Model Performance — Repeat Purchase Prediction:")
    print(pd.DataFrame(metrics).set_index("Model").to_string())

    winner = pd.DataFrame(metrics).set_index("Model")["F1-Score"].idxmax()
    print(f"\n🏆 Best Model  : {winner}")

    print("""
📁 Output Files
   output/ecommerce_orders.csv        ← Raw dataset
   output/rfm_segments.csv            ← Customer RFM table
   output/charts/
     01_monthly_revenue_trend.png     ← Revenue over time
     02_category_analysis.png         ← Category KPIs
     03_region_category_heatmap.png   ← Revenue by region
     04_payment_shipping.png          ← Payment & shipping
     05_discount_impact.png           ← Discount analysis
     06_customer_age_analysis.png     ← Age group insights
     07_rfm_segments.png              ← Customer segments
     08_prediction_confusion.png      ← ML confusion matrix
     09_prediction_roc.png            ← ML ROC curves
""")
    print("🎉 Task 4 Complete! Push to GitHub and submit.")
