# ============================================================
# main.py
# Task 2 - Predictive Modeling (Thiranex Internship)
# ============================================================
# 🚀 SINGLE ENTRY POINT — just run this file!
#    python main.py
#
# WHAT THIS DOES (end to end):
#   1. Generates a Customer Churn dataset
#   2. Preprocesses it (encode → split → scale)
#   3. Trains Logistic Regression + Decision Tree
#   4. Evaluates both models with metrics
#   5. Generates 5 professional visualizations
#   6. Prints a final summary with the winner
# ============================================================

from generate_dataset import create_churn_dataset
from preprocessing    import preprocess_pipeline
from models           import (
    train_logistic_regression,
    train_decision_tree,
    evaluate_model,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_decision_tree_structure,
    plot_model_comparison,
)
import pandas as pd


def print_banner(text: str) -> None:
    print("\n" + "╔" + "═" * 53 + "╗")
    print(f"║  {text:<51}║")
    print("╚" + "═" * 53 + "╝\n")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    print_banner("TASK 2 — Predictive Modeling Using ML")

    # ── STEP 1: Generate Dataset ──────────────────────────
    print_banner("STEP 1: Generate Customer Churn Dataset")
    df = create_churn_dataset(n=500)

    # NOTE: To use YOUR own dataset, comment the line above and use:
    # import pandas as pd
    # df = pd.read_csv("your_dataset.csv")

    # ── STEP 2: Preprocess ───────────────────────────────
    print_banner("STEP 2: Preprocess Data")
    X_train, X_test, y_train, y_test, feature_names, scaler = \
        preprocess_pipeline(df)

    # ── STEP 3: Train Models ─────────────────────────────
    print_banner("STEP 3: Train ML Models")
    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)

    # ── STEP 4: Evaluate Models ──────────────────────────
    print_banner("STEP 4: Evaluate Models")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")

    metrics_list = [lr_metrics, dt_metrics]

    # ── STEP 5: Visualizations ───────────────────────────
    print_banner("STEP 5: Generate Visualizations")

    models_dict = {
        "Logistic Regression": lr_model,
        "Decision Tree"      : dt_model,
    }

    plot_confusion_matrices(models_dict, X_test, y_test)
    plot_roc_curves(models_dict, X_test, y_test)
    plot_feature_importance(dt_model, feature_names)
    plot_decision_tree_structure(dt_model, feature_names)
    plot_model_comparison(metrics_list)

    # ── STEP 6: Final Summary ────────────────────────────
    print_banner("FINAL SUMMARY")

    df_results = pd.DataFrame(metrics_list).set_index("Model")
    print(df_results.to_string())

    # Determine winner by F1-Score (best metric for imbalanced classes)
    winner = df_results["F1-Score"].idxmax()
    winner_f1 = df_results.loc[winner, "F1-Score"]

    print(f"\n🏆 Best Model : {winner}")
    print(f"   F1-Score   : {winner_f1}%")
    print(f"   (F1-Score used as tiebreaker — best for churn detection)")

    print("\n📁 All charts saved to: output/charts/")
    print("   01_confusion_matrices.png")
    print("   02_roc_curves.png")
    print("   03_feature_importance.png")
    print("   04_decision_tree_structure.png")
    print("   05_model_comparison.png")

    print("\n🎉 Task 2 Complete! You can now submit this to Thiranex.")
