# ============================================================
# models.py
# Task 2 - Predictive Modeling (Thiranex Internship)
# Train, evaluate & compare ML models
# ============================================================
# MODELS WE USE:
#
#  1. Logistic Regression
#     → The "linear" model for classification.
#     → Learns a straight-line decision boundary.
#     → Fast, interpretable, good baseline.
#     → Think: "If monthly_charges is high AND tenure is low → likely churn"
#
#  2. Decision Tree Classifier
#     → Learns by asking YES/NO questions about features.
#     → Like a flowchart: "tenure < 12?" → yes → "support_calls > 5?" → ...
#     → Easy to visualize and explain to non-technical people.
#     → Can overfit if tree is too deep (we limit max_depth to fix this)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier, plot_tree
from sklearn.metrics       import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13,
                     "axes.titleweight": "bold"})

OUTPUT_DIR = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# SECTION 1 : TRAIN BOTH MODELS
# ══════════════════════════════════════════════

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    max_iter=1000  → give it enough iterations to converge
    C=1.0          → regularization strength (higher = less regularization)
    """
    print("=" * 55)
    print("  MODEL 1 : LOGISTIC REGRESSION — Training …")
    print("=" * 55)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train, y_train)

    print("   ✔ Logistic Regression trained successfully!\n")
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    max_depth=5     → limits tree depth to prevent overfitting
    min_samples_split=10 → a node needs ≥10 samples to split further
    """
    print("=" * 55)
    print("  MODEL 2 : DECISION TREE — Training …")
    print("=" * 55)

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("   ✔ Decision Tree trained successfully!")
    print(f"   Tree depth: {model.get_depth()}  |  "
          f"Leaves: {model.get_n_leaves()}\n")
    return model


# ══════════════════════════════════════════════
# SECTION 2 : EVALUATE A SINGLE MODEL
# ══════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Calculates 4 key classification metrics:

    Accuracy  → out of ALL predictions, how many were correct?
    Precision → of all predicted "churned", how many actually churned?
                (low precision = many false alarms)
    Recall    → of all ACTUAL churned customers, how many did we catch?
                (low recall = missing real churners — costly for business!)
    F1-Score  → harmonic mean of precision & recall (best overall metric)
    """
    print(f"{'─'*55}")
    print(f"  EVALUATION: {model_name}")
    print(f"{'─'*55}")

    y_pred = model.predict(X_test)

    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall"   : round(recall_score(y_test, y_pred) * 100, 2),
        "F1-Score" : round(f1_score(y_test, y_pred) * 100, 2),
        "AUC-ROC"  : round(roc_auc_score(y_test,
                           model.predict_proba(X_test)[:, 1]) * 100, 2),
    }

    print(f"\n   Accuracy  : {metrics['Accuracy']}%")
    print(f"   Precision : {metrics['Precision']}%")
    print(f"   Recall    : {metrics['Recall']}%")
    print(f"   F1-Score  : {metrics['F1-Score']}%")
    print(f"   AUC-ROC   : {metrics['AUC-ROC']}%")

    print(f"\n📋 Full Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                 target_names=["Stayed (0)", "Churned (1)"]))
    return metrics


# ══════════════════════════════════════════════
# SECTION 3 : VISUALIZATIONS
# ══════════════════════════════════════════════

# ── VIZ 1: Side-by-side Confusion Matrices ──

def plot_confusion_matrices(models: dict, X_test, y_test) -> None:
    """
    A confusion matrix shows 4 outcomes:
      TP (True Positive)  → predicted churn,  actually churned  ✅
      TN (True Negative)  → predicted stayed, actually stayed   ✅
      FP (False Positive) → predicted churn,  actually stayed   ❌
      FN (False Negative) → predicted stayed, actually churned  ❌ (worst!)
    """
    print("\n📊 Plotting Confusion Matrices …")

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    colors = ["Blues", "Greens"]

    for ax, (name, model), cmap in zip(axes, models.items(), colors):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        # Annotate TP, TN, FP, FN
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75, labels[i][j],
                        ha="center", va="center",
                        fontsize=9, color="grey", style="italic")

    plt.suptitle("Confusion Matrices — Model Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    path = f"{OUTPUT_DIR}/01_confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ── VIZ 2: ROC Curves ──

def plot_roc_curves(models: dict, X_test, y_test) -> None:
    """
    ROC Curve plots True Positive Rate vs False Positive Rate.
    The closer the curve hugs the top-left corner, the better.
    AUC (Area Under Curve): 1.0 = perfect, 0.5 = random guessing.
    """
    print("📈 Plotting ROC Curves …")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = ["#2d6be4", "#27ae60"]

    for (name, model), color in zip(models.items(), colors):
        y_prob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)

        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{name}  (AUC = {auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)

    # Random baseline (diagonal)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline (AUC = 0.5)")

    ax.set_title("ROC Curves — Model Comparison")
    ax.set_xlabel("False Positive Rate  (1 - Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    sns.despine()

    path = f"{OUTPUT_DIR}/02_roc_curves.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ── VIZ 3: Feature Importance (Decision Tree) ──

def plot_feature_importance(dt_model, feature_names: list) -> None:
    """
    Decision Trees can tell us WHICH features matter most.
    Higher importance = that feature was used more for splitting.
    """
    print("🌳 Plotting Feature Importances (Decision Tree) …")

    importances = pd.Series(
        dt_model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2d6be4" if v >= importances.max() * 0.5
              else "#a8c7f5" for v in importances]

    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(ax.patches, importances):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_title("Feature Importances — Decision Tree")
    ax.set_xlabel("Importance Score")
    ax.set_xlim([0, importances.max() * 1.2])

    high_patch = mpatches.Patch(color="#2d6be4",  label="High importance (≥50% of max)")
    low_patch  = mpatches.Patch(color="#a8c7f5", label="Lower importance")
    ax.legend(handles=[high_patch, low_patch], loc="lower right")
    sns.despine()

    path = f"{OUTPUT_DIR}/03_feature_importance.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ── VIZ 4: Decision Tree Structure ──

def plot_decision_tree_structure(dt_model, feature_names: list) -> None:
    """
    Visualizes the actual decision tree — the flowchart of
    YES/NO questions the model uses to predict churn.
    """
    print("🌲 Plotting Decision Tree Structure (depth=3) …")

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        dt_model,
        feature_names = feature_names,
        class_names   = ["Stayed", "Churned"],
        filled        = True,
        rounded       = True,
        fontsize      = 9,
        max_depth     = 3,    # show only top 3 levels (readable)
        ax            = ax,
        impurity      = False,
    )
    ax.set_title("Decision Tree Structure (Top 3 Levels)", fontsize=14,
                 fontweight="bold", pad=15)

    path = f"{OUTPUT_DIR}/04_decision_tree_structure.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ── VIZ 5: Model Comparison Bar Chart ──

def plot_model_comparison(metrics_list: list) -> None:
    """
    Side-by-side bar chart comparing all metrics for both models.
    Makes it easy to pick the better model at a glance.
    """
    print("📊 Plotting Model Comparison Chart …")

    df_metrics = pd.DataFrame(metrics_list).set_index("Model")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    df_plot = df_metrics[metric_cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_cols))
    width = 0.35

    colors = ["#2d6be4", "#27ae60"]
    for i, (model_name, row) in enumerate(df_plot.iterrows()):
        bars = ax.bar(x + i * width, row.values, width,
                      label=model_name, color=colors[i],
                      edgecolor="white", alpha=0.88)
        # Value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim([0, 115])
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.3, label="Random baseline")
    sns.despine()

    path = f"{OUTPUT_DIR}/05_model_comparison.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")
