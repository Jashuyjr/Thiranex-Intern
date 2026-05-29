# ============================================================
# analysis.py
# Task 4 - Real-world Data Project : Retail E-Commerce
# Thiranex Internship
# ============================================================
# COVERS:
#   1. Data cleaning & feature engineering
#   2. Business KPI calculations
#   3. EDA visualizations (revenue trends, categories, regions)
#   4. Customer segmentation (RFM analysis)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                      "axes.titleweight": "bold"})

OUTPUT_DIR = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INR = lambda x, _: f"₹{x/1000:.0f}K"     # y-axis formatter helper


# ══════════════════════════════════════════════
# STEP 1 : CLEAN & ENGINEER FEATURES
# ══════════════════════════════════════════════

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data and adds derived columns used in analysis.
    """
    print("=" * 58)
    print("  STEP 1 : DATA CLEANING & FEATURE ENGINEERING")
    print("=" * 58)

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    # ── Derived time columns ──
    df["month"]      = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"]    = df["order_date"].dt.quarter.map(
                           {1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    df["week"]       = df["order_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]= df["order_date"].dt.day_name()

    # ── Revenue after returns ──
    df["net_revenue"] = df["total_amount"] * (1 - df["returned"])

    # ── Profit proxy (assume 30% margin before discount) ──
    df["profit"]      = np.round(df["net_revenue"] * 0.30
                                 - df["discount_amt"] * df["quantity"], 2)

    # ── Age buckets ──
    df["age_group"] = pd.cut(df["customer_age"],
                              bins=[0,25,35,45,55,70],
                              labels=["18-24","25-34","35-44","45-54","55+"])

    # ── Discount tier ──
    df["discount_tier"] = pd.cut(df["discount_pct"],
                                  bins=[-1,0,10,20,30],
                                  labels=["No Discount","Low (1-10%)","Mid (11-20%)","High (21-30%)"])

    print(f"   ✔ Missing values : {df.isnull().sum().sum()}")
    print(f"   ✔ Duplicates     : {df.duplicated().sum()}")
    print(f"   ✔ New columns    : month, quarter, net_revenue, profit, age_group, discount_tier")
    print(f"   ✔ Final shape    : {df.shape}\n")
    return df


# ══════════════════════════════════════════════
# STEP 2 : BUSINESS KPIs
# ══════════════════════════════════════════════

def print_business_kpis(df: pd.DataFrame) -> None:
    """
    Prints the key business metrics every e-commerce report needs.
    """
    print("=" * 58)
    print("  STEP 2 : BUSINESS KPIs")
    print("=" * 58)

    total_revenue  = df["total_amount"].sum()
    net_revenue    = df["net_revenue"].sum()
    total_orders   = len(df)
    unique_cust    = df["customer_id"].nunique()
    avg_order_val  = df["total_amount"].mean()
    return_rate    = df["returned"].mean() * 100
    repeat_rate    = df["repeat_purchase"].mean() * 100
    avg_discount   = df["discount_pct"].mean()
    avg_rating     = df["rating"].mean()
    top_category   = df.groupby("category")["total_amount"].sum().idxmax()
    top_region     = df.groupby("region")["total_amount"].sum().idxmax()
    revenue_lost   = total_revenue - net_revenue

    print(f"\n  {'Metric':<30} {'Value':>15}")
    print(f"  {'─'*45}")
    print(f"  {'Total Gross Revenue':<30} {'₹'+f'{total_revenue:,.0f}':>15}")
    print(f"  {'Net Revenue (after returns)':<30} {'₹'+f'{net_revenue:,.0f}':>15}")
    print(f"  {'Revenue Lost to Returns':<30} {'₹'+f'{revenue_lost:,.0f}':>15}")
    print(f"  {'Total Orders':<30} {total_orders:>15,}")
    print(f"  {'Unique Customers':<30} {unique_cust:>15,}")
    print(f"  {'Avg Order Value':<30} {'₹'+f'{avg_order_val:,.0f}':>15}")
    print(f"  {'Return Rate':<30} {return_rate:>14.1f}%")
    print(f"  {'Repeat Purchase Rate':<30} {repeat_rate:>14.1f}%")
    print(f"  {'Avg Discount Given':<30} {avg_discount:>14.1f}%")
    print(f"  {'Avg Customer Rating':<30} {avg_rating:>14.2f}/5")
    print(f"  {'Top Category (Revenue)':<30} {top_category:>15}")
    print(f"  {'Top Region (Revenue)':<30} {top_region:>15}")
    print()


# ══════════════════════════════════════════════
# STEP 3 : EDA VISUALIZATIONS
# ══════════════════════════════════════════════

def plot_monthly_revenue(df: pd.DataFrame) -> None:
    """CHART 1 — Monthly Revenue Trend (Gross vs Net)"""
    print("📈 Chart 1: Monthly Revenue Trend …")

    monthly = df.groupby("month").agg(
        gross=("total_amount","sum"),
        net  =("net_revenue",  "sum")
    ).reset_index()

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["label"] = monthly["month"].apply(lambda m: month_labels[m-1])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly["label"], monthly["gross"]/1e6, marker="o",
            lw=2.2, color="#2d6be4", label="Gross Revenue")
    ax.plot(monthly["label"], monthly["net"]/1e6,   marker="s",
            lw=2.2, color="#27ae60", linestyle="--", label="Net Revenue")
    ax.fill_between(monthly["label"], monthly["gross"]/1e6,
                    monthly["net"]/1e6, alpha=0.12, color="#e84545",
                    label="Return Loss")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"₹{x:.1f}M"))
    ax.set_title("Monthly Revenue Trend — Gross vs Net (2024)")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹ Millions)")
    ax.legend(); sns.despine()

    path = f"{OUTPUT_DIR}/01_monthly_revenue_trend.png"
    plt.tight_layout(); plt.savefig(path); plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_category_analysis(df: pd.DataFrame) -> None:
    """CHART 2 — Revenue, Orders & Return Rate by Category"""
    print("📦 Chart 2: Category Analysis …")

    cat = df.groupby("category").agg(
        revenue    =("total_amount",  "sum"),
        orders     =("order_id",      "count"),
        avg_rating =("rating",        "mean"),
        return_rate=("returned",      "mean"),
    ).reset_index().sort_values("revenue", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Category Performance Analysis", fontsize=14, fontweight="bold")

    # Revenue bar
    colors = sns.color_palette("Set2", len(cat))
    bars = axes[0].bar(cat["category"], cat["revenue"]/1e6,
                       color=colors, edgecolor="white")
    axes[0].set_title("Revenue by Category")
    axes[0].set_ylabel("Revenue (₹ Millions)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"₹{x:.1f}M"))
    axes[0].set_xticklabels(cat["category"], rotation=20, ha="right")
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"₹{bar.get_height():.1f}M", ha="center", fontsize=8)

    # Return rate bar
    ret_colors = ["#e84545" if r > 0.12 else "#2d6be4"
                  for r in cat["return_rate"]]
    bars2 = axes[1].bar(cat["category"], cat["return_rate"]*100,
                        color=ret_colors, edgecolor="white")
    axes[1].set_title("Return Rate by Category")
    axes[1].set_ylabel("Return Rate (%)")
    axes[1].set_xticklabels(cat["category"], rotation=20, ha="right")
    axes[1].axhline(df["returned"].mean()*100, color="grey",
                    linestyle="--", linewidth=1.2, label="Avg")
    axes[1].legend(fontsize=9)
    for bar in bars2:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=8)

    # Avg rating
    axes[2].barh(cat["category"], cat["avg_rating"],
                 color=sns.color_palette("YlGn", len(cat))[::-1],
                 edgecolor="white")
    axes[2].set_title("Avg Rating by Category")
    axes[2].set_xlabel("Rating (out of 5)")
    axes[2].set_xlim([0, 5.5])
    axes[2].axvline(4.0, color="grey", linestyle="--", linewidth=1)
    for i, v in enumerate(cat["avg_rating"]):
        axes[2].text(v+0.05, i, f"{v:.2f}", va="center", fontsize=9)

    sns.despine()
    path = f"{OUTPUT_DIR}/02_category_analysis.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_regional_revenue(df: pd.DataFrame) -> None:
    """CHART 3 — Revenue Heatmap: Region × Category"""
    print("🗺️  Chart 3: Regional Revenue Heatmap …")

    pivot = (df.groupby(["region","category"])["total_amount"]
               .sum().div(1e6).round(2).unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label":"Revenue (₹ Millions)"}, ax=ax)
    ax.set_title("Revenue Heatmap — Region × Category (₹ Millions)")
    ax.set_xlabel("Category"); ax.set_ylabel("Region")
    plt.xticks(rotation=20, ha="right"); plt.yticks(rotation=0)

    path = f"{OUTPUT_DIR}/03_region_category_heatmap.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_payment_and_shipping(df: pd.DataFrame) -> None:
    """CHART 4 — Payment Method & Shipping preference breakdown"""
    print("💳 Chart 4: Payment & Shipping Analysis …")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Payment Method & Shipping Preferences", fontsize=14,
                 fontweight="bold")

    # Payment donut
    pay_counts = df["payment_method"].value_counts()
    wedges, texts, autotexts = axes[0].pie(
        pay_counts, labels=pay_counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set3", len(pay_counts)),
        startangle=90, pctdistance=0.80,
        wedgeprops={"edgecolor":"white","linewidth":2})
    # Make it a donut
    centre = plt.Circle((0,0), 0.55, fc="white")
    axes[0].add_patch(centre)
    axes[0].set_title("Payment Method Share")

    # Shipping grouped bar: method vs avg shipping days + % orders
    ship = df.groupby("shipping_method").agg(
        orders      =("order_id","count"),
        avg_days    =("shipping_days","mean"),
        avg_revenue =("total_amount","mean"),
    ).reset_index()

    x = np.arange(len(ship))
    w = 0.35
    b1 = axes[1].bar(x-w/2, ship["avg_days"], w, color="#2d6be4",
                     label="Avg Shipping Days", edgecolor="white")
    ax2= axes[1].twinx()
    b2 = ax2.bar(x+w/2, ship["orders"],  w, color="#f39c12",
                 label="Total Orders", edgecolor="white", alpha=0.8)

    axes[1].set_xticks(x); axes[1].set_xticklabels(ship["shipping_method"])
    axes[1].set_ylabel("Avg Shipping Days", color="#2d6be4")
    ax2.set_ylabel("Total Orders",          color="#f39c12")
    axes[1].set_title("Shipping Method — Days vs Order Volume")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1+lines2, labels1+labels2, loc="upper right",
                   fontsize=9)

    path = f"{OUTPUT_DIR}/04_payment_shipping.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_discount_impact(df: pd.DataFrame) -> None:
    """CHART 5 — Does discount drive higher sales & repeat purchases?"""
    print("🏷️  Chart 5: Discount Impact Analysis …")

    disc = df.groupby("discount_tier").agg(
        avg_order_val  =("total_amount",    "mean"),
        repeat_rate    =("repeat_purchase", "mean"),
        return_rate    =("returned",        "mean"),
        orders         =("order_id",        "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Impact of Discount on Customer Behaviour",
                 fontsize=14, fontweight="bold")

    tier_order = ["No Discount","Low (1-10%)","Mid (11-20%)","High (21-30%)"]
    disc = disc.set_index("discount_tier").reindex(tier_order).reset_index()

    colors = ["#95a5a6","#3498db","#27ae60","#e74c3c"]

    for ax, col, title, ylabel, fmt in zip(
        axes,
        ["avg_order_val","repeat_rate","return_rate"],
        ["Avg Order Value","Repeat Purchase Rate","Return Rate"],
        ["₹ Amount","%","%"],
        ["₹{:.0f}","{:.1%}","{:.1%}"]
    ):
        vals = disc[col] * (100 if col in ["repeat_rate","return_rate"] else 1)
        bars = ax.bar(disc["discount_tier"], vals, color=colors, edgecolor="white")
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticklabels(disc["discount_tier"], rotation=18, ha="right")
        if col != "avg_order_val":
            ax.set_ylim(0, 100)
        for bar in bars:
            label = (f"₹{bar.get_height():,.0f}" if col=="avg_order_val"
                     else f"{bar.get_height():.1f}%")
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    label, ha="center", fontsize=8.5)

    sns.despine()
    path = f"{OUTPUT_DIR}/05_discount_impact.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


def plot_customer_age_analysis(df: pd.DataFrame) -> None:
    """CHART 6 — Revenue & Repeat Purchase by Age Group"""
    print("👥 Chart 6: Customer Age Group Analysis …")

    age = df.groupby("age_group", observed=True).agg(
        revenue     =("total_amount",    "sum"),
        orders      =("order_id",        "count"),
        repeat_rate =("repeat_purchase", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Customer Age Group Analysis", fontsize=14,
                 fontweight="bold")

    colors = sns.color_palette("Blues_d", len(age))[::-1]

    # Revenue by age
    bars = axes[0].bar(age["age_group"].astype(str), age["revenue"]/1e6,
                       color=colors, edgecolor="white")
    axes[0].set_title("Revenue by Age Group")
    axes[0].set_ylabel("Revenue (₹ Millions)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"₹{x:.1f}M"))
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"₹{bar.get_height():.1f}M", ha="center", fontsize=9)

    # Repeat purchase rate by age
    axes[1].plot(age["age_group"].astype(str), age["repeat_rate"]*100,
                 marker="o", lw=2.5, color="#e74c3c", markersize=8)
    axes[1].fill_between(age["age_group"].astype(str),
                         age["repeat_rate"]*100, alpha=0.15, color="#e74c3c")
    axes[1].set_title("Repeat Purchase Rate by Age Group")
    axes[1].set_ylabel("Repeat Purchase Rate (%)")
    axes[1].set_ylim(0, 100)
    for i, (ag, rr) in enumerate(zip(age["age_group"], age["repeat_rate"])):
        axes[1].text(i, rr*100+1.5, f"{rr*100:.1f}%", ha="center", fontsize=9)

    sns.despine()
    path = f"{OUTPUT_DIR}/06_customer_age_analysis.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


# ══════════════════════════════════════════════
# STEP 4 : RFM CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════

def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM = Recency, Frequency, Monetary — classic customer segmentation.

    Recency   → How recently did they order? (lower = better)
    Frequency → How many orders total?       (higher = better)
    Monetary  → How much did they spend?     (higher = better)

    Segments:
      Champions    → bought recently, buy often, spend most
      Loyal        → buy often, good spenders
      At Risk      → used to buy often but haven't recently
      Lost         → haven't bought in a long time
    """
    print("=" * 58)
    print("  STEP 4 : RFM CUSTOMER SEGMENTATION")
    print("=" * 58)

    snapshot_date = pd.Timestamp("2025-01-01")

    rfm = df.groupby("customer_id").agg(
        recency  =("order_date",   lambda x: (snapshot_date - x.max()).days),
        frequency=("order_id",     "count"),
        monetary =("total_amount", "sum"),
    ).reset_index()

    # Score each metric 1–4 (quartile-based)
    rfm["R"] = pd.qcut(rfm["recency"],   4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"),
                       4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"],  4, labels=[1,2,3,4]).astype(int)
    rfm["RFM_Score"] = rfm["R"] + rfm["F"] + rfm["M"]

    # Segment labels
    def segment(score):
        if score >= 11: return "Champions"
        elif score >= 9: return "Loyal Customers"
        elif score >= 7: return "Potential Loyalists"
        elif score >= 5: return "At Risk"
        else:            return "Lost"

    rfm["segment"] = rfm["RFM_Score"].apply(segment)

    seg_summary = rfm.groupby("segment").agg(
        customers=("customer_id","count"),
        avg_recency=("recency","mean"),
        avg_frequency=("frequency","mean"),
        avg_monetary=("monetary","mean"),
    ).round(1)

    print("\n📊 RFM Segment Summary:")
    print(seg_summary.to_string())

    rfm.to_csv("output/rfm_segments.csv", index=False)
    print(f"\n   ✔ RFM table saved → output/rfm_segments.csv\n")
    return rfm


def plot_rfm_segments(rfm: pd.DataFrame) -> None:
    """CHART 7 — RFM Segment Distribution"""
    print("🎯 Chart 7: RFM Segment Distribution …")

    seg_order  = ["Champions","Loyal Customers","Potential Loyalists",
                  "At Risk","Lost"]
    seg_colors = ["#27ae60","#2d6be4","#f39c12","#e67e22","#e84545"]

    seg_counts = (rfm["segment"].value_counts()
                      .reindex(seg_order, fill_value=0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RFM Customer Segmentation", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(seg_counts.index, seg_counts.values,
                       color=seg_colors, edgecolor="white")
    axes[0].set_title("Customers per Segment")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_xticklabels(seg_counts.index, rotation=20, ha="right")
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(int(bar.get_height())), ha="center", fontsize=9,
                     fontweight="bold")

    # Scatter: Frequency vs Monetary, coloured by segment
    palette = dict(zip(seg_order, seg_colors))
    for seg, color in palette.items():
        sub = rfm[rfm["segment"]==seg]
        axes[1].scatter(sub["frequency"], sub["monetary"]/1000,
                        c=color, alpha=0.55, s=40, label=seg,
                        edgecolors="none")

    axes[1].set_title("Frequency vs Monetary (by Segment)")
    axes[1].set_xlabel("Order Frequency")
    axes[1].set_ylabel("Total Spend (₹ Thousands)")
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.7)
    sns.despine(ax=axes[1])

    path = f"{OUTPUT_DIR}/07_rfm_segments.png"
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"   ✔ Saved → {path}\n")


if __name__ == "__main__":
    from generate_dataset import create_ecommerce_dataset
    df_raw = create_ecommerce_dataset()

    df = clean_and_engineer(df_raw)
    print_business_kpis(df)

    print("─"*58 + "\n  STEP 3 : EDA VISUALIZATIONS\n" + "─"*58)
    plot_monthly_revenue(df)
    plot_category_analysis(df)
    plot_regional_revenue(df)
    plot_payment_and_shipping(df)
    plot_discount_impact(df)
    plot_customer_age_analysis(df)

    rfm = rfm_segmentation(df)
    plot_rfm_segments(rfm)

    print("🎉 Analysis complete! Run prediction.py next.")
