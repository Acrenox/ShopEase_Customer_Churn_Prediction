# E-Commerce Customer Churn Prediction & Retention

## Business Problem

Chroma, a fictional e-commerce client, is seeing declining repeat purchases. The marketing team needs to identify customers who are likely to churn so they can prioritize retention campaigns before customers become inactive.

This project converts transaction history into customer-level features, trains churn classification models, and turns model probabilities into practical retention segments.

## Dataset

The project uses the UCI Online Retail dataset:

https://archive.ics.uci.edu/dataset/352/online+retail

Citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.

License: Creative Commons Attribution 4.0 International (CC BY 4.0), as listed by the UCI Machine Learning Repository.

Expected raw file:

```text
data/raw/Online Retail.xlsx
```

The dataset contains transaction-level records with invoice, product, quantity, date, price, customer, and country fields.

## Architecture

```text
Raw Data
-> Cleaning
-> RFM Features
-> Engagement Features
-> Temporal Churn Label
-> ML Models
-> Evaluation
-> Risk Segmentation
-> Retention Strategy
```

## Project Files

- `main.py`: Runs the full workflow from data download to reports.
- `src/config.py`: Central paths, constants, dataset URLs, and directory setup.
- `src/data_loader.py`: Downloads, loads, and inspects the raw UCI dataset.
- `src/preprocessing.py`: Cleans transactions and creates revenue.
- `src/feature_engineering.py`: Creates temporal churn labels, RFM features, and engagement features without leakage.
- `src/train.py`: Builds, trains, cross-validates, and selects classification models.
- `src/evaluate.py`: Calculates precision, recall, F1-score, ROC-AUC, and evaluation plots.
- `src/feature_importance.py`: Extracts model feature importance.
- `src/segmentation.py`: Converts probabilities into risk and business segments.
- `src/reporting.py`: Creates EDA and business charts.
- `notebooks/`: Beginner-friendly notebooks for exploration, feature engineering, modeling, and segmentation.
- `IMPLEMENTATION.md`: Detailed implementation design.
- `WORKFLOW.md`: Step-by-step execution workflow.

## Features

- `Recency`: Days since the customer's most recent purchase before the cutoff date.
- `Frequency`: Number of unique invoices in the feature window.
- `Monetary`: Total customer revenue in the feature window.
- `AverageOrderValue`: Customer revenue divided by unique orders.
- `TotalItems`: Total purchased item quantity.
- `UniqueProducts`: Number of distinct products purchased.
- `CustomerLifetimeDays`: Days between first and most recent purchase in the feature window.
- `AverageDaysBetweenOrders`: Average gap between customer order dates.
- `UniqueInvoices`: Number of unique invoices.
- `PurchaseFrequency`: Orders per active customer lifetime day.
- `AverageItemsPerOrder`: Items purchased per order.
- `NumberOfActiveMonths`: Count of months where the customer purchased.
- `WeekendPurchaseRatio`: Share of purchases made on weekends.

## Churn Definition

The project uses a temporal churn label to prevent target leakage:

```text
Feature window:
Transactions up to the cutoff date.

Prediction window:
The 90 days after the cutoff date.

Churn = 1:
Customer made no purchase in the future prediction window.

Churn = 0:
Customer made at least one purchase in the future prediction window.
```

Features are calculated only from transactions before or on the cutoff date. Future transactions are used only for the churn label.

## Models

The workflow compares:

- Logistic Regression
- Random Forest
- XGBoost

The selected model is based on actual validation/test performance and the business objective. XGBoost is not automatically selected.

## Evaluation

The project evaluates models using:

- Precision
- Recall
- F1-score
- ROC-AUC

Accuracy is not the primary metric because churn data can be imbalanced. In churn prediction, recall is often important because false negatives are customers who may churn without being targeted by a retention campaign.

## Results

The full pipeline was run on the real UCI Online Retail dataset. The temporal cutoff date was `2011-09-10`, with the following 90 days used as the prediction window through `2011-12-09`.

Cleaning summary:

```text
Original rows: 541,909
Rows removed: 149,217
Final clean rows: 392,692
Unique clean customers: 4,338
Total clean revenue: 8,887,208.89
```

Feature dataset summary:

```text
Modeled customers: 3,370
Churn rate: 43.06%
Churned customers: 1,451
Non-churned customers: 1,919
```

Model comparison:

| Model | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5854 | 0.7448 | 0.6555 | 0.7337 |
| XGBoost | 0.5977 | 0.7276 | 0.6563 | 0.7299 |
| Random Forest | 0.5937 | 0.7103 | 0.6468 | 0.7101 |

Selected model:

```text
Logistic Regression
```

Logistic Regression was selected because it produced the highest ROC-AUC and strongest recall in this run, which fits the churn objective of identifying as many at-risk customers as practical.

Top feature importance drivers from the selected model:

| Feature | Importance |
| --- | ---: |
| NumberOfActiveMonths | 0.2512 |
| UniqueProducts | 0.1619 |
| AverageOrderValue | 0.1555 |
| Monetary | 0.1505 |
| Recency | 0.0843 |

Business summary:

```text
Total customers: 3,370
High-risk customers: 258
Medium-risk customers: 2,041
Low-risk customers: 1,071
Revenue at risk: 1,201,139.14
Average churn probability: 48.59%
```

Generated outputs:

```text
reports/model_metrics.csv
reports/feature_importance.csv
reports/customer_predictions.csv
reports/customer_risk_segments.csv
reports/business_summary.csv
models/churn_model.pkl
```

The metrics in `reports/model_metrics.csv` are calculated from actual model predictions. Do not edit them manually.

## Business Output

The selected model generates `ChurnProbability` for every customer. The project then assigns:

```text
High Risk:   probability >= 0.75
Medium Risk: 0.40 <= probability < 0.75
Low Risk:    probability < 0.40
```

It also calculates:

```text
RevenueAtRisk = Monetary * ChurnProbability
```

Business segments:

- `VIP At Risk`: High churn risk and high monetary value.
- `At Risk`: Medium/high churn risk and medium or high value.
- `Loyal Customers`: Low churn risk and high value.
- `Low Engagement`: Lower value customers needing automated re-engagement.

Recommendations are business rules, not ML predictions.

## How to Run

Create and activate an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the full workflow:

```bash
python main.py
```

Or with `uv`:

```bash
uv sync --cache-dir .uv-cache
uv run --cache-dir .uv-cache python main.py
```

Open notebooks:

```bash
jupyter notebook notebooks/
```

## Interview Explanation

The e-commerce client wanted to identify customers at risk of churn and improve retention. I used transaction-level purchase data to build customer-level RFM and engagement features, including recency, frequency, monetary value, order value, product diversity, and customer lifetime. I defined churn using a temporal prediction window so the model did not learn from future data. I compared Logistic Regression, Random Forest, and XGBoost using precision, recall, F1-score, and ROC-AUC, with attention to recall because missed churners reduce retention opportunities. The selected model generated churn probabilities, which I converted into risk segments and combined with customer value to prioritize targeted retention campaigns.
