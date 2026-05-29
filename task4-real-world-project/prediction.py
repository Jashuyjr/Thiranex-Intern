# ============================================================
# prediction.py
# Task 4 - Real-world Data Project : Retail E-Commerce
# Predict whether a customer will make a repeat purchase
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.metrics            import (accuracy_score, f1_score,
                                         roc_auc_score, confusion_matrix,
                                         classification_report, roc_curve)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                      "axes.titleweight": "bold"})
OUTPUT_DIR = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# STEP 1 : PREPARE FEATURES
# ══════════════════════════════════════════════

def prepare_features(df: pd.DataFrame):
    """
    Selects and encodes features for the prediction model.
    Target: repeat_purchase (1 = will buy again, 0 = won't)
    """
    print("=" * 58)
    print("  PREDICTION STEP 1 : PREPARE FEATURES")
    print("=" * 58)

    features = [
        "category", "discount_pct", "quantity", "total_amount",
        "shipping_method", "shipping_days", "region",
        "customer_age", "gender", "payment_method",
        "rating", "returned",
    ]
    target = "repeat_purchase"

    df2 = df[features + [target]].copy()

    # Encode categoricals
    le  = LabelEncoder()
    cat = ["category","shipping_method","region","gender","payment_method"]
    for col in cat:
        df2[col] = le.fit_transform(df2[col])
        print(f"   ✔ Encoded: '{col}'")

    X = df2[features]
    y = df2[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    print(f"\n   Train size : {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"   Repeat rate (train): {y_train.mean()*100:.1f}%")
    print(f"   Repeat rate (test) : {y_test.mean()*100:.1f}%\n")

    return X_train_sc, X_test_sc, y_train, y_test, list(X.columns)


# ══════════════════════════════════════════════
# STEP 2 : TRAIN MODELS
# ══════════════════════════════════════════════

def train_models(X_train, y_train):
    """Train Logistic Regression and Decision Tree."""
    print("=" * 58)
    print("  PREDICTION STEP 2 : TRAIN MODELS")
    print("=" * 58)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train, y_train)
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"\n   ✔ Logistic Regression trained  |  CV F1: {lr_cv:.3f}")

    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=15,
                                 random_state=42)
    dt.fit(X_train, y_train)
    dt_cv = cross_val_score(dt, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"   ✔ Decision Tree trained         |  CV F1: {dt_cv:.3f}\n")

    return {"Logistic Regression": lr, "Decision Tree": dt}


# ══════════════════════════════════════════════
# STEP 3 : EVALUATE & VISUALIZE
# ══════════════════════════════════════════════

def evaluate_and_plot(models: dict, X_test, y_test, feature_names) -> list:
    """Evaluate both models and produce 2 charts."""
    print("=" * 58)
    print("  PREDICTION STEP 3 : EVALUATE MODELS")
    print("=" * 58)

    metrics_list = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        metrics_list.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred)*100, 2),
            "F1-Score" : round(f1_score(y_test, y_pred)*100, 2),
            "AUC-ROC"  : round(roc_auc_score(y_test, y_prob)*100, 2),
        })
        print(f"\n  ── {name} ──")
        print(classification_report(y_test, y_pred,
                                     target_names=["No Repeat","Repeat"]))

    # ── Chart 8: Confusion matrices ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Repeat Purchase Prediction",
                 fontsize=14, fontweight="bold")
    for ax, (name, model), cmap in zip(axes, models.items(), ["Blues","Greens"]):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=["No Repeat","Repeat"],
                    yticklabels=["No Repeat","Repeat"],
                    linewidths=1, linecolor="white",
                    annot_kws={"size":15,"weight":"bold"}, ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    path = f"{OUTPUT_DIR}/08_prediction_confusion.png"
    plt.tight_layout(); plt.savefig(path); plt.show()
    print(f"\n   ✔ Chart 8 saved → {path}")

    # ── Chart 9: ROC curves ──
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2d6be4","#27ae60"]
    for (name, model), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{name}  (AUC={auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)

    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Random (AUC=0.5)")
    ax.set_title("ROC Curves — Repeat Purchase Prediction")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    sns.despine()

    path = f"{OUTPUT_DIR}/09_prediction_roc.png"
    plt.tight_layout(); plt.savefig(path); plt.show()
    print(f"   ✔ Chart 9 saved → {path}\n")

    return metrics_list


if __name__ == "__main__":
    import pandas as pd
    from generate_dataset import create_ecommerce_dataset
    from analysis         import clean_and_engineer

    df_raw = create_ecommerce_dataset()
    df     = clean_and_engineer(df_raw)

    X_tr, X_te, y_tr, y_te, feat_names = prepare_features(df)
    models   = train_models(X_tr, y_tr)
    metrics  = evaluate_and_plot(models, X_te, y_te, feat_names)

    print("📊 Final Results:")
    print(pd.DataFrame(metrics).set_index("Model").to_string())
    print("\n🎉 Prediction module complete!")
