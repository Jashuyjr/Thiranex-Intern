# ============================================================
# visualizations.py
# Task 1 - Data Cleaning & Visualization (Thiranex Internship)
# Author  : [Your Name]
# Date    : May 2026
# ============================================================
# HOW TO RUN:
#   python visualizations.py
# (Make sure data_cleaning.py ran first so cleaned_data.csv exists)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ──────────────────────────────────────────────
# GLOBAL STYLE SETTINGS
# Apply once here so all plots look consistent
# ──────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titlesize"  : 14,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 12,
})

OUTPUT_DIR = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_clean_data(path: str = "output/cleaned_data.csv") -> pd.DataFrame:
    """Load the cleaned dataset produced by data_cleaning.py."""
    df = pd.read_csv(path)

    # Re-parse dates and categoricals after CSV round-trip
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
    if "age_group" in df.columns:
        order = ["<25", "25-34", "35-44", "45-64", "65+"]
        df["age_group"] = pd.Categorical(df["age_group"], categories=order, ordered=True)

    print(f"✅ Loaded cleaned data: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


# ══════════════════════════════════════════════
# VISUALIZATION 1
# Bar Chart — Total Revenue by Product Category
# WHY: Shows which products earn the most money.
#      Great for business decisions.
# ══════════════════════════════════════════════

def plot_revenue_by_product(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: total revenue per product,
    coloured by product category.
    """
    print("📊 Creating Visualization 1: Revenue by Product …")

    # Aggregate
    rev_df = (df.groupby(["product", "category"])["total_revenue"]
                .sum()
                .reset_index()
                .sort_values("total_revenue", ascending=False))

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data    = rev_df,
        x       = "product",
        y       = "total_revenue",
        hue     = "category",
        palette = "Set2",
        edgecolor = "white",
        ax      = ax,
    )

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
    )

    # Add value labels on top of each bar
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 500,
                f"₹{height/1000:.1f}K",
                ha="center", va="bottom", fontsize=9, color="#333"
            )

    ax.set_title("Total Revenue by Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Total Revenue")
    ax.legend(title="Category", loc="upper right")
    sns.despine()

    path = f"{OUTPUT_DIR}/01_revenue_by_product.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# VISUALIZATION 2
# Heatmap — Correlation Matrix of Numerical Columns
# WHY: Reveals relationships between variables.
#      e.g., Does higher price → lower rating?
# ══════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap of Pearson correlation coefficients
    for all numerical columns.
    """
    print("🌡️  Creating Visualization 2: Correlation Heatmap …")

    # Select only numeric columns for correlation
    num_df = df.select_dtypes(include=["float64", "int64"]).drop(
        columns=["order_id"], errors="ignore"
    )

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot      = True,      # show numbers inside cells
        fmt        = ".2f",     # 2 decimal places
        cmap       = "coolwarm",# red = positive, blue = negative
        center     = 0,         # white = no correlation
        linewidths = 0.5,
        linecolor  = "white",
        square     = True,
        ax         = ax,
    )

    ax.set_title("Correlation Matrix — Numerical Features")

    # Tilt x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = f"{OUTPUT_DIR}/02_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# VISUALIZATION 3
# Box Plot — Unit Price distribution by Region
# WHY: Compares price spread across regions and
#      makes outliers immediately visible.
# ══════════════════════════════════════════════

def plot_price_distribution_by_region(df: pd.DataFrame) -> None:
    """
    Side-by-side box plots: unit_price per region.
    Overlaid with a strip plot (individual dots) for detail.
    """
    print("📦 Creating Visualization 3: Price Distribution by Region …")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Box plot layer
    sns.boxplot(
        data      = df,
        x         = "region",
        y         = "unit_price",
        palette   = "pastel",
        width     = 0.5,
        fliersize = 0,           # hide default outlier dots (we use stripplot)
        ax        = ax,
    )

    # Strip plot layer — shows individual data points
    sns.stripplot(
        data      = df,
        x         = "region",
        y         = "unit_price",
        color     = "#2d6be4",
        alpha     = 0.35,
        size      = 3,
        jitter    = True,
        ax        = ax,
    )

    # Format y-axis
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}")
    )

    ax.set_title("Unit Price Distribution by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Unit Price (₹)")
    sns.despine()

    path = f"{OUTPUT_DIR}/03_price_by_region.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# BONUS VISUALIZATION 4
# Line Chart — Monthly Revenue Trend
# WHY: Shows sales growth/decline over time — 
#      a classic business KPI chart.
# ══════════════════════════════════════════════

def plot_monthly_revenue_trend(df: pd.DataFrame) -> None:
    """
    Line chart of total revenue aggregated by month.
    """
    if "sale_date" not in df.columns:
        print("⚠️  Skipping monthly trend — no 'sale_date' column found.\n")
        return

    print("📈 Creating Bonus Visualization 4: Monthly Revenue Trend …")

    monthly = (df.groupby(df["sale_date"].dt.to_period("M"))["total_revenue"]
                 .sum()
                 .reset_index())
    monthly["sale_date"] = monthly["sale_date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 4))

    # Main line
    ax.plot(
        monthly["sale_date"], monthly["total_revenue"],
        color="#2d6be4", linewidth=2.2, marker="o", markersize=5, label="Revenue"
    )

    # Shaded area below the line
    ax.fill_between(
        monthly["sale_date"], monthly["total_revenue"],
        alpha=0.12, color="#2d6be4"
    )

    # Format axes
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
    )
    fig.autofmt_xdate()

    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Revenue")
    ax.legend()
    sns.despine()

    path = f"{OUTPUT_DIR}/04_monthly_revenue_trend.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ──────────────────────────────────────────────
# MAIN — RUN ALL VISUALIZATIONS
# ──────────────────────────────────────────────

if __name__ == "__main__":

    df = load_clean_data("output/cleaned_data.csv")

    plot_revenue_by_product(df)
    plot_correlation_heatmap(df)
    plot_price_distribution_by_region(df)
    plot_monthly_revenue_trend(df)

    print("🎉 All visualizations saved to 'output/charts/' folder!")
    print("   Open each PNG to include in your project report.")
