"""EDA and business visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid")


def create_eda_figures(transactions: pd.DataFrame, features: pd.DataFrame) -> None:
    """Create the core EDA charts requested by the project brief."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    monthly = (
        transactions.set_index("InvoiceDate")
        .resample("ME")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .reset_index()
    )

    _lineplot(monthly, "InvoiceDate", "Revenue", "Revenue Over Time", "revenue_over_time.png")
    _lineplot(monthly, "InvoiceDate", "Orders", "Orders Over Time", "orders_over_time.png")
    _histplot(features, "Monetary", "Customer Revenue Distribution", "customer_revenue_distribution.png")
    _histplot(features, "Frequency", "Frequency Distribution", "frequency_distribution.png")
    _histplot(features, "Recency", "Recency Distribution", "recency_distribution.png")

    top_customers = features.nlargest(15, "Monetary")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_customers, x="Monetary", y=top_customers["CustomerID"].astype(str))
    plt.title("Top Customers by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("CustomerID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_customers_by_revenue.png", dpi=160)
    plt.close()

    corr = features.drop(columns=["CustomerID"]).corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="vlag", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def create_business_figures(segments: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    """Create charts for the business output section."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=segments, x="RiskSegment", order=["High Risk", "Medium Risk", "Low Risk"])
    plt.title("Risk Segment Distribution")
    plt.xlabel("Risk Segment")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_segment_distribution.png", dpi=160)
    plt.close()

    revenue_by_segment = (
        segments.groupby("RiskSegment", as_index=False)["RevenueAtRisk"].sum()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=revenue_by_segment, x="RiskSegment", y="RevenueAtRisk")
    plt.title("Revenue at Risk by Segment")
    plt.xlabel("Risk Segment")
    plt.ylabel("Revenue at Risk")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "revenue_at_risk_by_segment.png", dpi=160)
    plt.close()

    _histplot(
        segments,
        "ChurnProbability",
        "Churn Probability Distribution",
        "churn_probability_distribution.png",
    )

    top_at_risk = segments.nlargest(15, "RevenueAtRisk")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_at_risk, x="RevenueAtRisk", y=top_at_risk["CustomerID"].astype(str))
    plt.title("Top Customers by Revenue at Risk")
    plt.xlabel("Revenue at Risk")
    plt.ylabel("CustomerID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_customers_at_risk.png", dpi=160)
    plt.close()

    top_features = feature_importance.head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close()


def _lineplot(data: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x=x, y=y, marker="o")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def _histplot(data: pd.DataFrame, column: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], bins=40, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()

