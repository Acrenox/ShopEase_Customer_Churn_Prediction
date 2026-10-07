"""Data download, loading, and raw-data inspection utilities."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from src.config import RAW_DATA_PATH, UCI_DIRECT_XLSX_URL, UCI_ZIP_URL, ensure_directories


def download_dataset(output_path: Path = RAW_DATA_PATH) -> Path:
    """Download the UCI Online Retail dataset if it is not already present.

    The direct XLSX URL is attempted first because it saves the file exactly as
    the project expects it. The current UCI static ZIP URL is used as a fallback.
    """

    ensure_directories()
    if output_path.exists():
        print(f"Raw dataset already exists: {output_path}")
        return output_path

    try:
        _download_file(UCI_DIRECT_XLSX_URL, output_path)
        return output_path
    except requests.RequestException as exc:
        print(f"Direct XLSX download failed: {exc}")

    zip_path = output_path.with_suffix(".zip")
    _download_file(UCI_ZIP_URL, zip_path)
    with ZipFile(zip_path) as archive:
        matching_files = [
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        ]
        if not matching_files:
            raise FileNotFoundError("No XLSX file found inside downloaded UCI ZIP.")
        archive.extract(matching_files[0], output_path.parent)
        extracted_path = output_path.parent / matching_files[0]
        extracted_path.rename(output_path)
    zip_path.unlink(missing_ok=True)
    return output_path


def _download_file(url: str, output_path: Path) -> None:
    """Download a file with streaming so large files do not live in memory."""

    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
    print(f"Saved dataset to {output_path}")


def load_raw_transactions(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load transaction-level data from the raw Excel workbook."""

    return pd.read_excel(path, engine="openpyxl")


def summarize_raw_transactions(df: pd.DataFrame) -> dict[str, object]:
    """Return and print the checks requested in the workflow."""

    invoice_no = df["InvoiceNo"].astype(str)
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_customers": int(df["CustomerID"].nunique(dropna=True)),
        "date_min": pd.to_datetime(df["InvoiceDate"]).min(),
        "date_max": pd.to_datetime(df["InvoiceDate"]).max(),
        "cancelled_transactions": int(invoice_no.str.startswith("C", na=False).sum()),
    }

    print("Raw data summary")
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print("First rows:")
    print(df.head())
    print("Data types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isna().sum())
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Unique customers: {summary['unique_customers']}")
    print(f"Transaction date range: {summary['date_min']} to {summary['date_max']}")
    print(f"Cancelled transactions: {summary['cancelled_transactions']}")

    return summary

