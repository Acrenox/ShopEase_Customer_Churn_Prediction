# Implementation Plan

## Project Overview

Build an end-to-end e-commerce customer churn prediction system for the fictional client Chroma. The system uses transaction-level purchase data to create customer-level behavioral features, define churn using a future inactivity window, train classification models, and produce business-ready retention segments.

Primary objective:

```text
Raw transactions
-> Cleaning
-> RFM and engagement features
-> Temporal churn label
-> Model training and evaluation
-> Churn probabilities
-> Risk segments
-> Retention recommendations
```

Do not fabricate metrics, customer counts, revenue, feature importance, or business impact. All results must be calculated from the real dataset.

## Dataset

Use the UCI Online Retail dataset where possible:

https://archive.ics.uci.edu/dataset/352/online+retail

Expected raw file:

```text
data/raw/Online Retail.xlsx
```

Expected columns:

```text
InvoiceNo
StockCode
Description
Quantity
InvoiceDate
UnitPrice
CustomerID
Country
```

Document the source and license in `README.md`.

## Target Project Structure

```text
.
├── README.md
├── IMPLEMENTATION.md
├── WORKFLOW.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   └── processed/
│       ├── cleaned_transactions.csv
│       └── customer_features.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_churn_modeling.ipynb
│   └── 04_customer_segmentation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── segmentation.py
├── models/
│   └── churn_model.pkl
└── reports/
    ├── figures/
    ├── model_metrics.csv
    ├── feature_importance.csv
    ├── customer_predictions.csv
    └── customer_risk_segments.csv
```

## Dependencies

Keep the stack simple and interview-friendly:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
openpyxl
jupyter
joblib
```

Only add `shap` if SHAP explainability is implemented.

## Phase 1: Data Loading

Implement in `src/data_loader.py`.

Responsibilities:

- Download or load `data/raw/Online Retail.xlsx`.
- Load the Excel file with pandas.
- Display shape, first rows, dtypes, missing values, duplicates, unique customers, transaction date range, and cancelled transactions.
- Keep the function reusable for notebooks and scripts.

Useful functions:

```python
download_dataset() -> Path
load_raw_transactions(path: str | Path) -> pd.DataFrame
summarize_raw_transactions(df: pd.DataFrame) -> dict
```

## Phase 2: Cleaning

Implement in `src/preprocessing.py`.

Cleaning rules:

- Remove rows with missing `CustomerID`.
- Remove cancelled transactions where `InvoiceNo` starts with `C`.
- Remove rows where `Quantity <= 0`.
- Remove rows where `UnitPrice <= 0`.
- Remove duplicate rows.
- Convert `InvoiceDate` to datetime.
- Create `Revenue = Quantity * UnitPrice`.

Save:

```text
data/processed/cleaned_transactions.csv
```

Print or return:

- Original row count
- Rows removed
- Final row count
- Unique customers
- Total revenue

## Phase 3: EDA

Implement in `notebooks/01_data_exploration.ipynb`.

Business questions:

- How many customers are present?
- How many transactions occurred?
- What is total revenue?
- What is average order value?
- Which countries have the most customers?
- Which customers generate the most revenue?
- What does transaction volume look like over time?
- How does purchase behavior vary across customers?

Recommended charts:

- Revenue over time
- Orders over time
- Customer revenue distribution
- Frequency distribution
- Recency distribution
- Top customers by revenue
- Correlation heatmap

Save important figures under:

```text
reports/figures/
```

## Phase 4: Temporal Churn Definition

Avoid target leakage. Features must come from the feature window only, and labels must come from the future prediction window only.

Recommended setup:

```text
FEATURE_WINDOW:
Historical transactions up to a cutoff date

PREDICTION_WINDOW:
The next 90 days after the cutoff date
```

Label:

```text
Churn = 1 if the customer makes no purchase during the next 90 days.
Churn = 0 if the customer makes at least one purchase during the next 90 days.
```

If the dataset date range cannot support a fixed cutoff, determine a valid cutoff automatically from the available dates.

## Phase 5: Feature Engineering

Implement in `src/feature_engineering.py`.

All features must be calculated only from the feature window.

Required RFM features:

- `Recency`: days since most recent purchase relative to cutoff date
- `Frequency`: number of unique invoices/orders
- `Monetary`: total revenue

Required engagement features:

- `AverageOrderValue`
- `TotalItems`
- `UniqueProducts`
- `CustomerLifetimeDays`
- `AverageDaysBetweenOrders`
- `UniqueInvoices`

Useful optional features:

- `PurchaseFrequency`
- `AverageItemsPerOrder`
- `NumberOfActiveMonths`
- `WeekendPurchaseRatio`

Save:

```text
data/processed/customer_features.csv
```

Validate:

- No duplicate customers
- No missing target values
- No infinite feature values
- Expected churn class distribution
- Features do not use future transactions

## Phase 6: Modeling Dataset

Prepare:

```text
X = customer features without CustomerID and Churn
y = Churn
```

Use a preprocessing pipeline where appropriate:

- Impute missing values if needed.
- Handle infinite values before model training.
- Scale numeric features for Logistic Regression.
- Use stratified train/test split when class balance allows it.

Do not use accuracy as the primary metric.

## Phase 7: Model Training

Implement in `src/train.py`.

Train and compare:

- Logistic Regression
- Random Forest
- XGBoost

Use:

- Reproducible random seed
- Stratified train/test split where appropriate
- Cross-validation on training data when practical
- No test-set tuning

For XGBoost, start with reasonable parameters and adjust only when supported by validation results:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `scale_pos_weight` only if class imbalance requires it

## Phase 8: Evaluation

Implement in `src/evaluate.py`.

Metrics:

- Precision
- Recall
- F1-score
- ROC-AUC

Save:

```text
reports/model_metrics.csv
```

Also generate:

- Confusion matrix
- ROC curve
- Classification report

Explain the business trade-off:

- False positives: customers incorrectly targeted for retention.
- False negatives: at-risk customers missed by the campaign.

For churn, recall is often important because missed churners represent lost retention opportunities.

## Phase 9: Model Selection

Select the final model based on validation performance and business objective. Do not automatically choose XGBoost.

Save the selected model:

```text
models/churn_model.pkl
```

Use `joblib` for persistence.

## Phase 10: Feature Importance

Calculate feature importance for the selected model.

Save:

```text
reports/feature_importance.csv
```

Only write business interpretations after the actual model has been trained. For example, if `Recency` is important, explain that customers who have not purchased recently are more likely to churn.

## Phase 11: Churn Probability and Risk Segments

Generate a churn probability for every customer:

```text
CustomerID
ChurnProbability
```

Save:

```text
reports/customer_predictions.csv
```

Risk segment rules:

```text
High Risk:   ChurnProbability >= 0.75
Medium Risk: 0.40 <= ChurnProbability < 0.75
Low Risk:    ChurnProbability < 0.40
```

These are business rules and can be adjusted after validation.

Save:

```text
reports/customer_risk_segments.csv
```

## Phase 12: Business Segmentation

Create:

```text
RevenueAtRisk = Monetary * ChurnProbability
```

Suggested segment logic:

- `VIP At Risk`: high churn probability and high monetary value
- `At Risk`: high or medium churn probability and medium value
- `Loyal Customers`: low churn probability and high value
- `Low Engagement`: low value and medium or high churn probability

Use percentile-based thresholds for customer value instead of arbitrary revenue cutoffs.

## Phase 13: Retention Recommendations

Map segments to business actions:

```text
VIP At Risk:
Personalized discount, priority support, exclusive offers, personalized recommendations

At Risk:
Discount campaign, free shipping, email reminder, product recommendations

Loyal Customers:
Loyalty rewards, cross-selling, early product access

Low Engagement:
Automated re-engagement campaign, newsletter, product discovery campaign
```

Recommendations are business rules, not ML predictions.

## Phase 14: README Results

Populate the README only after actual execution.

Include:

- Business problem
- Dataset source
- Architecture
- Feature explanations
- Model comparison table
- Selected model
- Business outputs
- How to run
- Interview explanation under 60 seconds

Do not include placeholder metrics as if they were real.

