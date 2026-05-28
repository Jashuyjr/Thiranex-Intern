# ============================================================
# eda_analysis.py
# Task 3 - Exploratory Data Analysis (Thiranex Internship)
# Dataset : Customer Churn (reused from Task 2)
# ============================================================
# WHAT IS EDA?
#   Before building any model, we need to UNDERSTAND the data.
#   EDA answers questions like:
#     → What does the data look like? (shape, types, stats)
#     → Are there patterns? (correlations, trends)
#     → What affects churn the most?
#     → Are there any surprises in the data?
#
# HOW TO RUN:
#   python eda_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Global Style ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi"      : 130,
    "axes.titlesize"  : 13,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 11,
})

# Colour palette — consistent across all charts
CHURN_PALETTE = {0: "#2d6be4", 1: "#e84545"}   # blue=stayed, red=churned
OUTPUT_DIR    = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# SECTION 1 : LOAD & GENERATE DATA
# ══════════════════════════════════════════════

def load_or_generate_data() -> pd.DataFrame:
    """
    Tries to load the churn CSV from Task 2 output.
    If not found, regenerates it automatically.
    """
    path = "../task2-predictive-modeling/output/churn_dataset.csv"

    if not os.path.exists(path):
        # Fallback: regenerate inline
        path = "output/churn_dataset.csv"
        os.makedirs("output", exist_ok=True)

        np.random.seed(42)
        n = 500
        tenure           = np.random.randint(1, 72, n)
        monthly_charges  = np.round(np.random.uniform(300, 2000, n), 2)
        total_charges    = np.round(tenure * monthly_charges
                                    * np.random.uniform(0.9, 1.1, n), 2)
        num_support_calls = np.random.randint(0, 10, n)
        age              = np.random.randint(18, 70, n)
        contract_type    = np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"], n,
            p=[0.55, 0.25, 0.20])
        internet_service = np.random.choice(
            ["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20])
        has_tech_support = np.random.choice(["Yes", "No"], n, p=[0.40, 0.60])

        churn_prob = (
            0.40 * (tenure < 12).astype(float)
          + 0.25 * (contract_type == "Month-to-Month")
          + 0.20 * (num_support_calls > 5).astype(float)
          + 0.10 * (monthly_charges > 1500).astype(float)
          - 0.15 * (has_tech_support == "Yes")
          - 0.10 * (contract_type == "Two Year")
        )
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
            "churn"            : churn,
        })
        df.to_csv(path, index=False)
        print("✅ Dataset regenerated (Task 2 output not found).")
    else:
        df = pd.read_csv(path)
        print(f"✅ Loaded dataset from Task 2: {path}")

    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


# ══════════════════════════════════════════════
# SECTION 2 : OVERVIEW & SUMMARY STATISTICS
# ══════════════════════════════════════════════

def print_data_overview(df: pd.DataFrame) -> None:
    """
    Prints a structured summary of the dataset:
      - Shape, dtypes, missing values
      - Numerical statistics (mean, std, min, max, quartiles)
      - Categorical value counts
      - Churn distribution
    """
    print("=" * 60)
    print("  STEP 1 : DATA OVERVIEW & SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\n📐 Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Duplicates : {df.duplicated().sum()}")
    print(f"❓ Missing    : {df.isnull().sum().sum()} total\n")

    # Data types
    print("📊 Column Info:")
    info = pd.DataFrame({
        "dtype"  : df.dtypes,
        "unique" : df.nunique(),
        "nulls"  : df.isnull().sum(),
        "null_%"  : (df.isnull().mean() * 100).round(2),
    })
    print(info.to_string())

    # Numerical summary
    print("\n📈 Numerical Summary:")
    num_df = df.select_dtypes(include="number").drop(
        columns=["churn"], errors="ignore")
    print(num_df.describe().round(2).to_string())

    # Categorical summary
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print("\n🏷️  Categorical Columns:")
    for col in cat_cols:
        if col == "customer_id":
            continue
        counts = df[col].value_counts()
        print(f"\n   {col}:")
        for val, cnt in counts.items():
            pct = cnt / len(df) * 100
            print(f"     {val:<20} {cnt:>4}  ({pct:.1f}%)")

    # Target distribution
    churn_counts = df["churn"].value_counts()
    print(f"\n🎯 Target — Churn Distribution:")
    print(f"   Stayed  (0): {churn_counts[0]}  ({churn_counts[0]/len(df)*100:.1f}%)")
    print(f"   Churned (1): {churn_counts[1]}  ({churn_counts[1]/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════
# SECTION 3 : UNIVARIATE ANALYSIS
# Looks at ONE variable at a time
# ══════════════════════════════════════════════

def plot_univariate_analysis(df: pd.DataFrame) -> None:
    """
    CHART 1 — Univariate Analysis

    For numerical columns: histogram + KDE curve
      Shows the DISTRIBUTION — is it normal? skewed? bimodal?

    For categorical columns: count bar chart
      Shows HOW MANY customers fall in each category
    """
    print("\n📊 Chart 1: Univariate Analysis …")

    num_cols = ["age", "tenure", "monthly_charges",
                "total_charges", "num_support_calls"]
    cat_cols = ["contract_type", "internet_service", "has_tech_support"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Univariate Analysis — Distribution of Each Feature",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── Numerical histograms ──
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        sns.histplot(df[col], kde=True, color="#2d6be4",
                     edgecolor="white", alpha=0.75, ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        # Annotate mean line
        mean_val = df[col].mean()
        ax.axvline(mean_val, color="red", linestyle="--",
                   linewidth=1.2, label=f"Mean: {mean_val:.1f}")
        ax.legend(fontsize=8)

    # ── Categorical bar charts ──
    for j, col in enumerate(cat_cols):
        ax = fig.add_subplot(gs[1 + j // 2, 2 + j % 2])
        order  = df[col].value_counts().index
        counts = df[col].value_counts()
        colors = sns.color_palette("Set2", len(order))
        bars   = ax.bar(order, counts[order], color=colors, edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    str(int(bar.get_height())),
                    ha="center", fontsize=8)

    # ── Churn pie chart (last cell) ──
    ax_pie = fig.add_subplot(gs[2, 3])
    churn_counts = df["churn"].value_counts()
    ax_pie.pie(
        churn_counts,
        labels     = ["Stayed", "Churned"],
        colors     = ["#2d6be4", "#e84545"],
        autopct    = "%1.1f%%",
        startangle = 90,
        wedgeprops = {"edgecolor": "white", "linewidth": 2},
    )
    ax_pie.set_title("Churn Split")

    path = f"{OUTPUT_DIR}/01_univariate_analysis.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# SECTION 4 : BIVARIATE ANALYSIS
# Looks at TWO variables — feature vs churn
# ══════════════════════════════════════════════

def plot_bivariate_numerical(df: pd.DataFrame) -> None:
    """
    CHART 2 — Numerical Features vs Churn (Box Plots)

    For each numerical column, compare its distribution
    between churned (1) and stayed (0) customers.

    KEY INSIGHT: If the boxes are at different heights,
    that feature is a good predictor of churn!
    """
    print("📦 Chart 2: Bivariate — Numerical vs Churn …")

    num_cols = ["age", "tenure", "monthly_charges",
                "total_charges", "num_support_calls"]

    fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 5))
    fig.suptitle("Bivariate Analysis — Numerical Features vs Churn",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, num_cols):
        sns.boxplot(
            data    = df,
            x       = "churn",
            y       = col,
            hue     = "churn",
            palette = {0: "#2d6be4", 1: "#e84545"},
            width   = 0.5,
            legend  = False,
            ax      = ax,
        )
        # Add mean markers
        means = df.groupby("churn")[col].mean()
        ax.scatter([0, 1], means.values, color="white",
                   s=60, zorder=5, label="Mean")

        ax.set_title(col.replace("_", " ").title())
        ax.set_xticklabels(["Stayed\n(0)", "Churned\n(1)"])
        ax.set_xlabel("")

        # Annotate means
        for x_pos, mean_val in enumerate(means.values):
            ax.text(x_pos, mean_val, f" {mean_val:.1f}",
                    va="center", fontsize=8, color="black")

    # Legend
    stayed_patch  = plt.matplotlib.patches.Patch(
        color="#2d6be4", label="Stayed (0)")
    churned_patch = plt.matplotlib.patches.Patch(
        color="#e84545", label="Churned (1)")
    fig.legend(handles=[stayed_patch, churned_patch],
               loc="upper right", bbox_to_anchor=(1, 1))

    sns.despine()
    path = f"{OUTPUT_DIR}/02_bivariate_numerical_vs_churn.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_bivariate_categorical(df: pd.DataFrame) -> None:
    """
    CHART 3 — Categorical Features vs Churn (Grouped Bar Charts)

    For each category, shows churn rate per group.
    e.g. Month-to-Month customers churn more than 2-Year contract ones.
    """
    print("📊 Chart 3: Bivariate — Categorical vs Churn …")

    cat_cols = ["contract_type", "internet_service", "has_tech_support"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Bivariate Analysis — Categorical Features vs Churn Rate",
                 fontsize=14, fontweight="bold")

    for ax, col in zip(axes, cat_cols):
        # Calculate churn rate per category
        churn_rate = (df.groupby(col)["churn"]
                        .mean()
                        .mul(100)
                        .sort_values(ascending=False)
                        .reset_index())
        churn_rate.columns = [col, "churn_rate"]

        colors = ["#e84545" if r > 30 else "#2d6be4"
                  for r in churn_rate["churn_rate"]]

        bars = ax.bar(churn_rate[col], churn_rate["churn_rate"],
                      color=colors, edgecolor="white", width=0.5)

        # Value labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%",
                    ha="center", fontsize=10, fontweight="bold")

        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xticklabels(churn_rate[col], rotation=15, ha="right")
        ax.axhline(y=df["churn"].mean() * 100, color="grey",
                   linestyle="--", linewidth=1.2, label="Avg churn rate")
        ax.set_ylim(0, 80)
        ax.legend(fontsize=8)

    # Colour legend
    high_patch = plt.matplotlib.patches.Patch(color="#e84545", label=">30% churn")
    low_patch  = plt.matplotlib.patches.Patch(color="#2d6be4", label="≤30% churn")
    fig.legend(handles=[high_patch, low_patch],
               loc="upper right", bbox_to_anchor=(1.01, 1))

    sns.despine()
    path = f"{OUTPUT_DIR}/03_bivariate_categorical_vs_churn.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# SECTION 5 : MULTIVARIATE ANALYSIS
# Looks at 3+ variables together
# ══════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    CHART 4 — Correlation Heatmap (Multivariate)

    Shows how ALL numerical variables relate to each other.
    Values close to +1 = strong positive correlation
    Values close to -1 = strong negative correlation
    Values near 0     = no relationship

    KEY INSIGHT: Which features correlate most with churn?
    """
    print("🌡️  Chart 4: Correlation Heatmap …")

    num_df = df.select_dtypes(include="number").drop(
        columns=["customer_id"], errors="ignore")
    corr = num_df.corr()

    # Mask upper triangle (avoid redundancy)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        mask       = mask,
        annot      = True,
        fmt        = ".2f",
        cmap       = "coolwarm",
        center     = 0,
        linewidths = 0.5,
        linecolor  = "white",
        square     = True,
        cbar_kws   = {"shrink": 0.8},
        ax         = ax,
    )
    ax.set_title("Correlation Heatmap — All Numerical Features + Churn",
                 pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Highlight churn row
    ax.add_patch(plt.Rectangle(
        (0, len(corr) - 1), len(corr), 1,
        fill=False, edgecolor="#e84545", lw=2.5,
        label="Churn correlations"
    ))
    ax.legend(loc="upper right", fontsize=9)

    path = f"{OUTPUT_DIR}/04_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_scatter_tenure_charges(df: pd.DataFrame) -> None:
    """
    CHART 5 — Scatter Plot: Tenure vs Monthly Charges (coloured by Churn)

    Multivariate: 3 variables at once
      X axis → tenure
      Y axis → monthly_charges
      Color  → churn (red=churned, blue=stayed)

    KEY INSIGHT: Do churners cluster in a specific region?
    (Short tenure + High charges = high churn risk?)
    """
    print("🔵 Chart 5: Scatter — Tenure vs Monthly Charges by Churn …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multivariate: Tenure vs Monthly Charges — Coloured by Churn",
                 fontsize=14, fontweight="bold")

    # Left: plain scatter
    for churn_val, label, color in [(0, "Stayed", "#2d6be4"),
                                     (1, "Churned", "#e84545")]:
        subset = df[df["churn"] == churn_val]
        axes[0].scatter(subset["tenure"], subset["monthly_charges"],
                        c=color, alpha=0.45, s=25, label=label,
                        edgecolors="none")

    axes[0].set_title("Scatter Plot")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Monthly Charges (₹)")
    axes[0].legend()
    sns.despine(ax=axes[0])

    # Right: KDE density contours per class
    for churn_val, color in [(0, "#2d6be4"), (1, "#e84545")]:
        subset = df[df["churn"] == churn_val]
        sns.kdeplot(
            data  = subset,
            x     = "tenure",
            y     = "monthly_charges",
            color = color,
            fill  = True,
            alpha = 0.25,
            ax    = axes[1],
            label = "Stayed" if churn_val == 0 else "Churned",
        )

    axes[1].set_title("KDE Density Contours")
    axes[1].set_xlabel("Tenure (months)")
    axes[1].set_ylabel("Monthly Charges (₹)")
    axes[1].legend()
    sns.despine(ax=axes[1])

    path = f"{OUTPUT_DIR}/05_scatter_tenure_charges.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_churn_by_contract_and_support(df: pd.DataFrame) -> None:
    """
    CHART 6 — Heatmap: Churn Rate by Contract Type × Support Calls Bucket

    Groups customers by their contract type AND how many
    support calls they made, then shows the churn rate for
    each combination — a 2D risk map.
    """
    print("🗺️  Chart 6: Churn Risk Map — Contract × Support Calls …")

    df2 = df.copy()
    # Bin support calls into 3 groups
    df2["support_bucket"] = pd.cut(
        df2["num_support_calls"],
        bins   = [-1, 2, 5, 9],
        labels = ["Low (0-2)", "Medium (3-5)", "High (6+)"]
    )

    pivot = (df2.groupby(["contract_type", "support_bucket"])["churn"]
               .mean()
               .mul(100)
               .round(1)
               .unstack())

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot,
        annot      = True,
        fmt        = ".1f",
        cmap       = "YlOrRd",
        linewidths = 0.5,
        linecolor  = "white",
        cbar_kws   = {"label": "Churn Rate (%)"},
        ax         = ax,
    )
    ax.set_title("Churn Rate (%) — Contract Type × Support Call Volume",
                 pad=12)
    ax.set_xlabel("Support Call Volume")
    ax.set_ylabel("Contract Type")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    path = f"{OUTPUT_DIR}/06_churn_risk_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_tenure_distribution_by_contract(df: pd.DataFrame) -> None:
    """
    CHART 7 — Violin Plot: Tenure Distribution by Contract Type

    Violin plots combine box plots + KDE density.
    Shows not just min/max/median but WHERE most values cluster.

    Split by churn (blue=stayed, red=churned) within each contract type.
    """
    print("🎻 Chart 7: Violin — Tenure by Contract Type + Churn …")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.violinplot(
        data       = df,
        x          = "contract_type",
        y          = "tenure",
        hue        = "churn",
        palette    = {0: "#2d6be4", 1: "#e84545"},
        split      = True,       # mirror the two halves
        inner      = "quartile", # show quartile lines inside
        ax         = ax,
    )

    ax.set_title("Tenure Distribution by Contract Type — Split by Churn")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Tenure (months)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Stayed (0)", "Churned (1)"], title="Churn",
              loc="upper right")
    sns.despine()

    path = f"{OUTPUT_DIR}/07_violin_tenure_by_contract.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# SECTION 6 : KEY INSIGHTS SUMMARY
# ══════════════════════════════════════════════

def print_key_insights(df: pd.DataFrame) -> None:
    """
    Prints data-driven insights derived from the EDA.
    These are the 5-6 sentences you write in your report.
    """
    print("\n" + "=" * 60)
    print("  STEP 6 : KEY INSIGHTS (Include These in Your Report)")
    print("=" * 60)

    # Insight 1 — Churn rate
    churn_rate = df["churn"].mean() * 100
    print(f"\n📌 Insight 1 — Overall Churn Rate")
    print(f"   {churn_rate:.1f}% of customers churned. "
          f"This is a {'high' if churn_rate > 25 else 'moderate'} churn rate "
          f"that warrants immediate attention.")

    # Insight 2 — Tenure difference
    avg_tenure_stayed  = df[df["churn"]==0]["tenure"].mean()
    avg_tenure_churned = df[df["churn"]==1]["tenure"].mean()
    print(f"\n📌 Insight 2 — Tenure vs Churn")
    print(f"   Churned customers had an average tenure of "
          f"{avg_tenure_churned:.1f} months vs "
          f"{avg_tenure_stayed:.1f} months for those who stayed.")
    print(f"   → NEW customers (short tenure) are most at risk.")

    # Insight 3 — Contract type
    churn_by_contract = (df.groupby("contract_type")["churn"]
                           .mean().mul(100).round(1)
                           .sort_values(ascending=False))
    top_contract = churn_by_contract.index[0]
    top_rate     = churn_by_contract.iloc[0]
    print(f"\n📌 Insight 3 — Contract Type")
    print(f"   '{top_contract}' customers have the highest churn rate "
          f"at {top_rate}%.")
    print(f"   → Offering longer contracts could significantly reduce churn.")

    # Insight 4 — Support calls
    high_support_churn = (df[df["num_support_calls"] > 5]["churn"].mean() * 100)
    low_support_churn  = (df[df["num_support_calls"] <= 2]["churn"].mean() * 100)
    print(f"\n📌 Insight 4 — Support Calls")
    print(f"   Customers with >5 support calls churn at {high_support_churn:.1f}% "
          f"vs only {low_support_churn:.1f}% for those with ≤2 calls.")
    print(f"   → High support call volume is a strong churn warning signal.")

    # Insight 5 — Tech support
    churn_with_support    = df[df["has_tech_support"]=="Yes"]["churn"].mean()*100
    churn_without_support = df[df["has_tech_support"]=="No"]["churn"].mean()*100
    print(f"\n📌 Insight 5 — Tech Support")
    print(f"   Customers WITH tech support churn at {churn_with_support:.1f}% "
          f"vs {churn_without_support:.1f}% without.")
    print(f"   → Tech support is a strong retention tool.")

    # Insight 6 — Top correlation with churn
    num_df = df.select_dtypes(include="number")
    corr_with_churn = (num_df.corr()["churn"]
                              .drop("churn")
                              .abs()
                              .sort_values(ascending=False))
    top_feature = corr_with_churn.index[0]
    top_corr    = corr_with_churn.iloc[0]
    print(f"\n📌 Insight 6 — Strongest Correlation with Churn")
    print(f"   '{top_feature}' has the highest correlation with churn "
          f"(r = {top_corr:.3f}).")
    print(f"   → This should be the primary feature in any predictive model.")

    print("\n" + "=" * 60)
    print("  ✅ Use these 6 insights in your EDA report / submission!")
    print("=" * 60)


# ══════════════════════════════════════════════
# MAIN — RUN COMPLETE EDA PIPELINE
# ══════════════════════════════════════════════

if __name__ == "__main__":

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   TASK 3 — Exploratory Data Analysis (EDA)          ║")
    print("║   Dataset : Customer Churn                          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Load Data ──────────────────────────────────────────
    df = load_or_generate_data()

    # ── Step 1: Overview ───────────────────────────────────
    print_data_overview(df)

    # ── Step 2: Univariate ─────────────────────────────────
    print("\n" + "─"*60)
    print("  STEP 2 : UNIVARIATE ANALYSIS")
    print("─"*60)
    plot_univariate_analysis(df)

    # ── Step 3: Bivariate ──────────────────────────────────
    print("─"*60)
    print("  STEP 3 : BIVARIATE ANALYSIS")
    print("─"*60)
    plot_bivariate_numerical(df)
    plot_bivariate_categorical(df)

    # ── Step 4: Multivariate ───────────────────────────────
    print("─"*60)
    print("  STEP 4 : MULTIVARIATE ANALYSIS")
    print("─"*60)
    plot_correlation_heatmap(df)
    plot_scatter_tenure_charges(df)
    plot_churn_by_contract_and_support(df)
    plot_tenure_distribution_by_contract(df)

    # ── Step 5: Key Insights ───────────────────────────────
    print_key_insights(df)

    print("\n📁 All 7 charts saved to: output/charts/")
    print("   01_univariate_analysis.png")
    print("   02_bivariate_numerical_vs_churn.png")
    print("   03_bivariate_categorical_vs_churn.png")
    print("   04_correlation_heatmap.png")
    print("   05_scatter_tenure_charges.png")
    print("   06_churn_risk_heatmap.png")
    print("   07_violin_tenure_by_contract.png")
    print("\n🎉 Task 3 Complete! Push to GitHub and submit.")
