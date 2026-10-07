"""Cleaning rules for the Online Retail transaction dataset."""

from __future__ import annotations

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean raw transactions and return the cleaned data plus validation stats."""

    original_rows = len(df)
    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["CustomerID"])
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str)
    cleaned = cleaned[~cleaned["InvoiceNo"].str.startswith("C", na=False)]
    cleaned = cleaned[cleaned["Quantity"] > 0]
    cleaned = cleaned[cleaned["UnitPrice"] > 0]
    cleaned = cleaned.drop_duplicates()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)
    cleaned["Revenue"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    stats = {
        "original_rows": original_rows,
        "rows_removed": original_rows - len(cleaned),
        "final_rows": len(cleaned),
        "unique_customers": cleaned["CustomerID"].nunique(),
        "total_revenue": float(cleaned["Revenue"].sum()),
    }
    return cleaned, stats


def print_cleaning_report(stats: dict[str, float]) -> None:
    """Print the required cleaning validation numbers."""

    print("Cleaning report")
    print(f"Original row count: {stats['original_rows']:,}")
    print(f"Rows removed: {stats['rows_removed']:,}")
    print(f"Final row count: {stats['final_rows']:,}")
    print(f"Unique customers: {stats['unique_customers']:,}")
    print(f"Total revenue: {stats['total_revenue']:,.2f}")

