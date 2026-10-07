"""Feature importance extraction for fitted sklearn pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract comparable feature importance values from the selected model."""

    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    total = importance["Importance"].sum()
    if total > 0:
        importance["Importance"] = importance["Importance"] / total
    return importance.sort_values("Importance", ascending=False)

