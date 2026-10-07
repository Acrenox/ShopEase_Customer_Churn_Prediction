"""Customer risk segmentation and retention recommendation rules."""

from __future__ import annotations

import pandas as pd

from src.evaluate import predict_positive_probability


def create_customer_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Generate churn probabilities for every customer."""

    X = features.drop(columns=["CustomerID", "Churn"])
    predictions = features[["CustomerID"]].copy()
    predictions["ChurnProbability"] = predict_positive_probability(model, X)
    return predictions


def assign_risk_segment(churn_probability: float) -> str:
    """Convert a probability into business risk buckets."""

    if churn_probability >= 0.75:
        return "High Risk"
    if churn_probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def build_customer_segments(
    features: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Combine model risk with customer value and recommended actions."""

    segments = features.merge(predictions, on="CustomerID", how="left")
    segments["RiskSegment"] = segments["ChurnProbability"].apply(assign_risk_segment)
    segments["RevenueAtRisk"] = segments["Monetary"] * segments["ChurnProbability"]

    high_value_cutoff = segments["Monetary"].quantile(0.75)
    medium_value_cutoff = segments["Monetary"].quantile(0.50)

    segments["BusinessSegment"] = segments.apply(
        lambda row: _business_segment(row, high_value_cutoff, medium_value_cutoff),
        axis=1,
    )
    segments["RetentionRecommendation"] = segments["BusinessSegment"].map(
        RETENTION_RECOMMENDATIONS
    )
    return segments.sort_values("RevenueAtRisk", ascending=False)


def _business_segment(row: pd.Series, high_value_cutoff: float, medium_value_cutoff: float) -> str:
    """Assign a practical retention segment from risk and customer value."""

    if row["RiskSegment"] == "High Risk" and row["Monetary"] >= high_value_cutoff:
        return "VIP At Risk"
    if row["RiskSegment"] in {"High Risk", "Medium Risk"} and row["Monetary"] >= medium_value_cutoff:
        return "At Risk"
    if row["RiskSegment"] == "Low Risk" and row["Monetary"] >= high_value_cutoff:
        return "Loyal Customers"
    return "Low Engagement"


RETENTION_RECOMMENDATIONS = {
    "VIP At Risk": (
        "Personalized discount; priority support; exclusive offers; "
        "personalized product recommendations"
    ),
    "At Risk": (
        "Discount campaign; free shipping; email reminder; product recommendations"
    ),
    "Loyal Customers": "Loyalty rewards; cross-selling; early product access",
    "Low Engagement": (
        "Automated re-engagement campaign; newsletter; product discovery campaign"
    ),
}


def create_business_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Create the dashboard-style summary requested in the workflow."""

    total_customers = len(segments)
    risk_counts = segments["RiskSegment"].value_counts()
    summary = {
        "Total Customers": total_customers,
        "Churn Rate": float(segments["Churn"].mean()),
        "High Risk Customers": int(risk_counts.get("High Risk", 0)),
        "Medium Risk Customers": int(risk_counts.get("Medium Risk", 0)),
        "Low Risk Customers": int(risk_counts.get("Low Risk", 0)),
        "Revenue At Risk": float(segments["RevenueAtRisk"].sum()),
        "Average Churn Probability": float(segments["ChurnProbability"].mean()),
    }
    return pd.DataFrame([summary])

