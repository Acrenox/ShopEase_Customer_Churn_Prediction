"""Run the complete customer churn prediction workflow.

This script is the reproducible path through the project. The notebooks explain
the same steps interactively, while this file creates the saved artifacts.
"""

from __future__ import annotations

from src.config import (
    BUSINESS_SUMMARY_PATH,
    CLEANED_TRANSACTIONS_PATH,
    CUSTOMER_FEATURES_PATH,
    CUSTOMER_PREDICTIONS_PATH,
    CUSTOMER_SEGMENTS_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    ensure_directories,
)
from src.data_loader import (
    download_dataset,
    load_raw_transactions,
    summarize_raw_transactions,
)
from src.evaluate import classification_report_frame, save_evaluation_figures
from src.feature_engineering import (
    build_customer_features,
    validate_customer_features,
)
from src.feature_importance import extract_feature_importance
from src.preprocessing import clean_transactions, print_cleaning_report
from src.reporting import create_business_figures, create_eda_figures
from src.segmentation import (
    build_customer_segments,
    create_business_summary,
    create_customer_predictions,
)
from src.train import save_model, train_and_select_model


def main() -> None:
    """Execute data acquisition, feature engineering, modeling, and reporting."""

    ensure_directories()

    print("\n1. Downloading/loading raw data")
    raw_path = download_dataset()
    raw_transactions = load_raw_transactions(raw_path)
    summarize_raw_transactions(raw_transactions)

    print("\n2. Cleaning transactions")
    cleaned_transactions, cleaning_stats = clean_transactions(raw_transactions)
    cleaned_transactions.to_csv(CLEANED_TRANSACTIONS_PATH, index=False)
    print_cleaning_report(cleaning_stats)
    print(f"Saved cleaned transactions to {CLEANED_TRANSACTIONS_PATH}")

    print("\n3. Building leakage-safe customer features")
    customer_features, feature_metadata = build_customer_features(cleaned_transactions)
    customer_features.to_csv(CUSTOMER_FEATURES_PATH, index=False)
    print(f"Feature cutoff date: {feature_metadata['cutoff_date'].date()}")
    print(f"Prediction window end: {feature_metadata['prediction_window_end'].date()}")
    print(f"Churn rate: {feature_metadata['churn_rate']:.2%}")
    print(validate_customer_features(customer_features))
    print(f"Saved customer features to {CUSTOMER_FEATURES_PATH}")

    print("\n4. Creating EDA figures")
    create_eda_figures(cleaned_transactions, customer_features)
    print(f"Saved EDA figures to {FIGURES_DIR}")

    print("\n5. Training and evaluating models")
    selected_model, metrics, context = train_and_select_model(customer_features)
    metrics.to_csv(MODEL_METRICS_PATH, index=False)
    print(metrics.to_string(index=False))
    print(f"Selected model: {context['selected_model_name']}")
    print(f"Saved metrics to {MODEL_METRICS_PATH}")

    report = classification_report_frame(
        selected_model, context["X_test"], context["y_test"]
    )
    report.to_csv(FIGURES_DIR.parent / "classification_report.csv")
    save_evaluation_figures(
        selected_model, context["X_test"], context["y_test"], FIGURES_DIR
    )

    print("\n6. Saving final model")
    save_model(selected_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    print("\n7. Creating feature importance and customer risk outputs")
    feature_importance = extract_feature_importance(
        selected_model, context["feature_names"]
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    predictions = create_customer_predictions(selected_model, customer_features)
    predictions.to_csv(CUSTOMER_PREDICTIONS_PATH, index=False)

    segments = build_customer_segments(customer_features, predictions)
    segments.to_csv(CUSTOMER_SEGMENTS_PATH, index=False)

    business_summary = create_business_summary(segments)
    business_summary.to_csv(BUSINESS_SUMMARY_PATH, index=False)
    create_business_figures(segments, feature_importance)

    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved customer predictions to {CUSTOMER_PREDICTIONS_PATH}")
    print(f"Saved customer risk segments to {CUSTOMER_SEGMENTS_PATH}")
    print(f"Saved business summary to {BUSINESS_SUMMARY_PATH}")
    print("\nWorkflow complete.")


if __name__ == "__main__":
    main()
