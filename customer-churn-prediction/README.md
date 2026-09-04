# Customer Churn Prediction

## Research Question
Which customers are most likely to churn (cancel service), and which demographic and service-related factors (contract type, tenure, monthly charge, payment method) can predict this behavior, so a telecom provider can target retention efforts effectively?

## Data
A 10,000-record telecom customer dataset (`churn_clean.csv`), covering contract type, tenure, monthly charges, internet service, payment method, and whether the customer churned. Not included in this repo (coursework-provided data); swap in your own copy with the same schema to reproduce.

## Approach
Three classification models were built and compared:

| Model | Accuracy | Notes |
|---|---|---|
| **Decision Tree** (max depth 5, on Contract/Tenure/MonthlyCharge) | **87%** | Best-performing model |
| **Logistic Regression** (reduced via Recursive Feature Elimination, 10 features) | **86%** | Improved from 81% pre-selection |
| Naive Bayes (Gaussian, full feature set) | 74% | Highest recall for actual churners (89%); reported from original coursework analysis, notebook not preserved |

## Key Finding
Contract length is the dominant churn driver: month-to-month customers have roughly **20-23x higher odds of churning** than customers on one- or two-year contracts (logistic regression coefficients of -3.02 and -3.14 respectively, converted to odds ratios). Tenure and monthly charge are secondary but still meaningful predictors.

## Files
- `logistic_regression.py` — data prep, initial model, RFE-based feature selection, reduced model
- `decision_tree.py` — data prep, decision tree fit, accuracy/MSE, feature importances

## Recommendation
Prioritize retention offers (contract incentives, loyalty pricing) for month-to-month, short-tenure, high-monthly-charge customers, the segment the models consistently flag as highest risk.
