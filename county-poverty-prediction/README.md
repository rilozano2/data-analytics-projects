# County-Level Poverty Prediction

## Research Question
Which economic and employment factors most strongly predict poverty rate across U.S. counties?

## Data
U.S. Census ACS 2017 county-level data (`acs2017_county_data.csv`), 3,220 counties, covering income, unemployment, employment-sector composition, and commute patterns. Source: [Kaggle — US Census Demographic Data](https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data).

## Approach
A multiple linear regression predicted county `Poverty` rate from income, unemployment rate, employment-sector shares (professional, service, construction, production), and commute variables.

- **R² = 0.79 (training), 0.72 (held-out test data)**
- Checked for multicollinearity via Variance Inflation Factor (VIF); dropped `IncomePerCap` (redundant with `Income`) before finalizing the model.

## Key Finding
Unemployment rate is the single strongest predictor of poverty, holding other factors constant: each 1-point increase in county unemployment rate corresponds to roughly a **0.9-point increase in poverty rate**. Higher median income and a larger professional-sector employment share are both independently associated with lower poverty.

## Limitation
Employment-sector and commute-mode variables are compositional (they sum to ~100% by construction), so some residual multicollinearity remains even after removing the most redundant variable, most notably in the `Drive` feature. This doesn't affect the model's overall predictive power (R²), but individual coefficients for the remaining collinear features should be read with that caveat.

## Files
- `poverty_regression.py` — data load, correlation check, OLS regression, VIF diagnostic, refit
