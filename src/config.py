"""Central configuration for paths and modeling constants.

Keeping these values in one file makes the workflow easier to explain and
prevents notebooks, scripts, and modules from silently drifting apart.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "Online Retail.xlsx"
CLEANED_TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "cleaned_transactions.csv"
CUSTOMER_FEATURES_PATH = PROCESSED_DATA_DIR / "customer_features.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
CUSTOMER_PREDICTIONS_PATH = REPORTS_DIR / "customer_predictions.csv"
CUSTOMER_SEGMENTS_PATH = REPORTS_DIR / "customer_risk_segments.csv"
BUSINESS_SUMMARY_PATH = REPORTS_DIR / "business_summary.csv"

RANDOM_SEED = 42
PREDICTION_WINDOW_DAYS = 90

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"
UCI_DIRECT_XLSX_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/"
    "Online%20Retail.xlsx"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""

    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

