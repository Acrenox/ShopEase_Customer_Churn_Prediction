# Project Workflow

This workflow describes the order for implementing and running the e-commerce churn prediction project.

## 1. Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

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

## 2. Create Project Directories

Create the working structure:

```bash
mkdir -p data/raw data/processed notebooks src models reports/figures
```

Expected outputs by the end of the project:

```text
data/processed/cleaned_transactions.csv
data/processed/customer_features.csv
models/churn_model.pkl
reports/model_metrics.csv
reports/feature_importance.csv
reports/customer_predictions.csv
reports/customer_risk_segments.csv
```

## 3. Acquire Data

Download the UCI Online Retail dataset:

```text
https://archive.ics.uci.edu/dataset/352/online+retail
```

Save it as:

```text
data/raw/Online Retail.xlsx
```

If automatic download fails, download the dataset manually from the UCI page and place it in the same path. Do not use synthetic data unless the real dataset is unavailable.

## 4. Inspect Raw Data

Run the data loading script or the first notebook section to check:

- Shape
- First rows
- Data types
- Missing values
- Duplicate rows
- Unique customers
- Transaction date range
- Cancelled transactions

Expected outcome:

```text
The project confirms that the raw file contains transaction-level purchase records with customer, product, quantity, date, price, and country fields.
```

## 5. Clean Transactions

Run preprocessing to:

- Remove missing `CustomerID`.
- Remove cancelled invoices.
- Remove non-positive quantities.
- Remove non-positive prices.
- Remove duplicates.
- Convert `InvoiceDate` to datetime.
- Create `Revenue`.

Save:

```text
data/processed/cleaned_transactions.csv
```

Record these validation numbers:

- Original row count
- Rows removed
- Final row count
- Unique customers
- Total revenue

## 6. Explore the Data

Use `notebooks/01_data_exploration.ipynb`.

Answer:

- How many customers are present?
- How many transactions occurred?
- What is total revenue?
- What is average order value?
- Which countries have the most customers?
- Which customers generate the most revenue?
- What does transaction volume look like over time?
- How does purchase behavior vary across customers?

Save useful charts under:

```text
reports/figures/
```

Avoid charts that do not support a business or modeling decision.

## 7. Define Feature and Prediction Windows

Use a temporal setup to prevent target leakage.

Example:

```text
Feature window:
All transactions up to cutoff date

Prediction window:
The 90 days after cutoff date
```

Define churn:

```text
Churn = 1 when a customer has no purchase in the prediction window.
Churn = 0 when a customer has at least one purchase in the prediction window.
```

Validation checklist:

- The cutoff date leaves enough history for features.
- The cutoff date leaves enough future data for the prediction window.
- No feature uses transactions after the cutoff date.
- The label is not used as an input feature.

## 8. Build Customer Features

Use `notebooks/02_feature_engineering.ipynb` and `src/feature_engineering.py`.

Create RFM features:

- `Recency`
- `Frequency`
- `Monetary`

Create engagement features:

- `AverageOrderValue`
- `TotalItems`
- `UniqueProducts`
- `CustomerLifetimeDays`
- `AverageDaysBetweenOrders`
- `UniqueInvoices`

Add the churn label.

Save:

```text
data/processed/customer_features.csv
```

Validation checklist:

- One row per customer
- No duplicate `CustomerID`
- No missing `Churn`
- No infinite feature values
- Reasonable churn and non-churn percentages

## 9. Train Models

Use `notebooks/03_churn_modeling.ipynb` and `src/train.py`.

Train:

- Logistic Regression
- Random Forest
- XGBoost

Recommended workflow:

```text
Load customer_features.csv
-> Split X and y
-> Remove CustomerID from X
-> Train/test split with stratification when possible
-> Fit preprocessing pipelines
-> Train candidate models
-> Compare validation or test metrics
```

Do not tune on the test set.

## 10. Evaluate Models

Calculate:

- Precision
- Recall
- F1-score
- ROC-AUC

Save:

```text
reports/model_metrics.csv
```

Also save or display:

- Confusion matrix
- ROC curve
- Classification report

Selection rule:

```text
Choose the model that best supports the business objective and validation performance.
```

Do not automatically choose XGBoost. If Random Forest or Logistic Regression performs better, select it and explain why.

## 11. Save Final Model

Save the selected model:

```text
models/churn_model.pkl
```

Use:

```python
joblib.dump(model, "models/churn_model.pkl")
```

The saved object should include preprocessing if preprocessing is required at prediction time.

## 12. Generate Feature Importance

Create feature importance for the selected model.

Save:

```text
reports/feature_importance.csv
```

Interpret only what the trained model actually shows. Do not claim specific drivers until the importance values are calculated.

## 13. Predict Churn Probability

Generate customer-level probabilities:

```text
CustomerID
ChurnProbability
```

Save:

```text
reports/customer_predictions.csv
```

The probability should represent the estimated probability that the customer will churn.

## 14. Segment Customers

Apply business thresholds:

```text
High Risk:   probability >= 0.75
Medium Risk: 0.40 <= probability < 0.75
Low Risk:    probability < 0.40
```

Calculate:

```text
RevenueAtRisk = Monetary * ChurnProbability
```

Create business segments:

- `VIP At Risk`
- `At Risk`
- `Loyal Customers`
- `Low Engagement`

Save:

```text
reports/customer_risk_segments.csv
```

## 15. Add Retention Actions

Map each customer segment to recommended actions:

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

These actions are business recommendations, not model predictions.

## 16. Create Business Summary

Create a summary with:

- Total customers
- Churn rate
- High-risk customers
- Medium-risk customers
- Low-risk customers
- Revenue at risk
- Average churn probability

Recommended charts:

- Risk segment distribution
- Revenue at risk by segment
- Churn probability distribution
- Top customers at risk
- Feature importance

A Streamlit dashboard is optional. Keep the core project focused on the ML pipeline.

## 17. Update README

Update `README.md` after the pipeline has been executed.

Include:

- Project title
- Business problem
- Dataset source and license
- Architecture
- Feature definitions
- Model comparison table
- Final model choice
- Actual metrics
- Business outputs
- How to run
- Short interview explanation

Do not write fake results. If metrics have not been generated yet, state that results will be populated after running the pipeline.

## 18. Final Quality Checklist

Before considering the project complete, verify:

- The real dataset is used.
- Cleaning outputs are saved.
- Feature engineering uses only the feature window.
- Churn labels use only the prediction window.
- The churn target is not included in model features.
- Metrics are calculated from actual predictions.
- The final model is saved.
- Reports are generated.
- README results match generated files.
- The project can run from a fresh environment.

## 19. Interview Story

Use this concise explanation:

```text
The e-commerce client wanted to identify customers at risk of churn and improve retention. I used transaction-level purchase data to build customer-level RFM and engagement features, including recency, frequency, monetary value, order value, product diversity, and customer lifetime. I defined churn using a temporal prediction window so the model did not learn from future data. I compared Logistic Regression, Random Forest, and XGBoost using precision, recall, F1-score, and ROC-AUC, with special attention to recall because missed churners reduce retention opportunities. The selected model generated churn probabilities, which I converted into risk segments and combined with customer value to prioritize targeted retention campaigns.
```

