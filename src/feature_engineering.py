"""Temporal churn labeling and customer-level feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PREDICTION_WINDOW_DAYS


def determine_cutoff_date(
    transactions: pd.DataFrame, prediction_window_days: int = PREDICTION_WINDOW_DAYS
) -> pd.Timestamp:
    """Choose the latest cutoff that leaves a full future prediction window."""

    max_date = transactions["InvoiceDate"].max().normalize()
    min_date = transactions["InvoiceDate"].min().normalize()
    cutoff_date = max_date - pd.Timedelta(days=prediction_window_days)
    if cutoff_date <= min_date:
        raise ValueError("Dataset does not contain enough time for a prediction window.")
    return cutoff_date


def split_feature_and_prediction_windows(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split transactions into historical features and future labels."""

    cutoff_date = cutoff_date or determine_cutoff_date(transactions, prediction_window_days)
    prediction_end = cutoff_date + pd.Timedelta(days=prediction_window_days)

    feature_window = transactions[transactions["InvoiceDate"] <= cutoff_date].copy()
    prediction_window = transactions[
        (transactions["InvoiceDate"] > cutoff_date)
        & (transactions["InvoiceDate"] <= prediction_end)
    ].copy()
    return feature_window, prediction_window, cutoff_date


def build_customer_features(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build leakage-safe customer features and a future-window churn label."""

    feature_window, prediction_window, cutoff_date = split_feature_and_prediction_windows(
        transactions, cutoff_date, prediction_window_days
    )

    invoice_level = (
        feature_window.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            InvoiceDate=("InvoiceDate", "min"),
            OrderRevenue=("Revenue", "sum"),
            OrderItems=("Quantity", "sum"),
            UniqueProductsInOrder=("StockCode", "nunique"),
        )
        .sort_values(["CustomerID", "InvoiceDate"])
    )

    grouped = feature_window.groupby("CustomerID")
    order_grouped = invoice_level.groupby("CustomerID")

    features = pd.DataFrame(index=grouped.size().index)
    features["Recency"] = (cutoff_date - grouped["InvoiceDate"].max()).dt.days
    features["Frequency"] = grouped["InvoiceNo"].nunique()
    features["Monetary"] = grouped["Revenue"].sum()
    features["AverageOrderValue"] = features["Monetary"] / features["Frequency"]
    features["TotalItems"] = grouped["Quantity"].sum()
    features["UniqueProducts"] = grouped["StockCode"].nunique()
    features["CustomerLifetimeDays"] = (
        grouped["InvoiceDate"].max() - grouped["InvoiceDate"].min()
    ).dt.days + 1
    features["UniqueInvoices"] = features["Frequency"]
    features["AverageItemsPerOrder"] = features["TotalItems"] / features["Frequency"]
    features["PurchaseFrequency"] = (
        features["Frequency"] / features["CustomerLifetimeDays"].clip(lower=1)
    )
    features["NumberOfActiveMonths"] = grouped["InvoiceDate"].apply(
        lambda dates: dates.dt.to_period("M").nunique()
    )
    features["WeekendPurchaseRatio"] = grouped["InvoiceDate"].apply(
        lambda dates: dates.dt.dayofweek.isin([5, 6]).mean()
    )

    avg_days = order_grouped["InvoiceDate"].apply(_average_days_between_orders)
    features["AverageDaysBetweenOrders"] = avg_days.reindex(features.index).fillna(0)

    customers_with_future_purchase = set(prediction_window["CustomerID"].unique())
    features["Churn"] = [
        0 if customer_id in customers_with_future_purchase else 1
        for customer_id in features.index
    ]

    features = (
        features.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "CustomerID"})
    )

    metadata = {
        "cutoff_date": cutoff_date,
        "prediction_window_days": prediction_window_days,
        "prediction_window_end": cutoff_date + pd.Timedelta(days=prediction_window_days),
        "feature_window_rows": len(feature_window),
        "prediction_window_rows": len(prediction_window),
        "customers": len(features),
        "churn_rate": float(features["Churn"].mean()),
    }
    return features, metadata


def _average_days_between_orders(order_dates: pd.Series) -> float:
    """Average gap between unique order dates for one customer."""

    unique_dates = order_dates.sort_values().dt.normalize().drop_duplicates()
    if len(unique_dates) < 2:
        return 0.0
    return float(unique_dates.diff().dt.days.dropna().mean())


def validate_customer_features(features: pd.DataFrame) -> dict[str, object]:
    """Return simple validation checks for the final modeling table."""

    numeric = features.drop(columns=["CustomerID"], errors="ignore")
    return {
        "rows": len(features),
        "duplicate_customers": int(features["CustomerID"].duplicated().sum()),
        "missing_values": features.isna().sum().to_dict(),
        "infinite_values": int(np.isinf(numeric.select_dtypes(include=[np.number])).sum().sum()),
        "churn_count": int(features["Churn"].sum()),
        "non_churn_count": int((features["Churn"] == 0).sum()),
        "churn_rate": float(features["Churn"].mean()),
    }

