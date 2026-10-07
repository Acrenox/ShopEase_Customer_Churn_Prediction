"""Model training, comparison, and model persistence."""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PATH, RANDOM_SEED
from src.evaluate import evaluate_classifier


def prepare_modeling_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the customer table into model features and churn target."""

    X = features.drop(columns=["CustomerID", "Churn"])
    y = features["Churn"].astype(int)
    return X, y


def build_candidate_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Create the required candidate classifiers."""

    class_counts = y_train.value_counts()
    imbalance_ratio = _xgboost_scale_pos_weight(class_counts)

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=imbalance_ratio,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        warnings.warn("xgboost is not installed; skipping XGBoost.", stacklevel=2)

    return models


def train_and_select_model(
    features: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, dict[str, object]]:
    """Train required models, compare metrics, and return the selected model."""

    X, y = prepare_modeling_data(features)
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    models = build_candidate_models(y_train)
    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        test_metrics = evaluate_classifier(model, X_test, y_test)
        cv_metrics = cross_validate_model(model, X_train, y_train)
        rows.append({"Model": model_name, **test_metrics, **cv_metrics})

    metrics = pd.DataFrame(rows).sort_values(
        by=["ROC-AUC", "F1", "Recall"], ascending=False, na_position="last"
    )
    selected_name = str(metrics.iloc[0]["Model"])
    selected_model = fitted_models[selected_name]

    context = {
        "selected_model_name": selected_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }
    return selected_model, metrics, context


def cross_validate_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run training-set cross-validation when class counts make it valid."""

    min_class_count = int(y_train.value_counts().min())
    if y_train.nunique() < 2 or min_class_count < 3:
        return {
            "CV_F1_Mean": np.nan,
            "CV_ROC_AUC_Mean": np.nan,
        }

    cv_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        n_jobs=-1,
    )
    return {
        "CV_F1_Mean": float(scores["test_f1"].mean()),
        "CV_ROC_AUC_Mean": float(scores["test_roc_auc"].mean()),
    }


def save_model(model: Pipeline, path=MODEL_PATH) -> None:
    """Persist the fitted preprocessing-plus-model pipeline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _xgboost_scale_pos_weight(class_counts: pd.Series) -> float:
    """Return negative/positive ratio for XGBoost imbalance handling."""

    positives = class_counts.get(1, 1)
    negatives = class_counts.get(0, 1)
    if positives == 0:
        return 1.0
    return float(negatives / positives)

