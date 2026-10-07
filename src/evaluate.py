"""Evaluation helpers for churn classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate the required churn metrics for a fitted classifier."""

    y_pred = model.predict(X_test)
    y_proba = predict_positive_probability(model, X_test)
    return {
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": _safe_roc_auc(y_test, y_proba),
    }


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities, falling back to scores when needed."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X)


def _safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """ROC-AUC is undefined when the test set has only one class."""

    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_proba)


def classification_report_frame(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return sklearn's classification report as a DataFrame."""

    y_pred = model.predict(X_test)
    return pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T


def save_evaluation_figures(model, X_test: pd.DataFrame, y_test: pd.Series, figures_dir) -> None:
    """Save confusion matrix, ROC, and precision-recall plots."""

    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    if len(set(y_test)) > 1:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(figures_dir / "roc_curve.png", dpi=160)
        plt.close()

        PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        plt.title("Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(figures_dir / "precision_recall_curve.png", dpi=160)
        plt.close()

