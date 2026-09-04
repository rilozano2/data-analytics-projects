import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('acs2017_county_data.csv')
features = ['Income','IncomePerCap','Unemployment','Professional','Service','Construction','Production','MeanCommute','Drive']
df = df.dropna(subset=features + ['Poverty'])
X = df[features]

vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("VIF (before):\n", vif.sort_values('VIF', ascending=False))

# Drop IncomePerCap (redundant with Income) and refit
features2 = ['Income','Unemployment','Professional','Service','Construction','Production','MeanCommute','Drive']
X2 = df[features2]
y = df['Poverty']

vif2 = pd.DataFrame()
vif2['feature'] = X2.columns
vif2['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
print("\nVIF (after dropping IncomePerCap):\n", vif2.sort_values('VIF', ascending=False))

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
X_train_sm = sm.add_constant(X_train)
model2 = sm.OLS(y_train, X_train_sm).fit()
X_test_sm = sm.add_constant(X_test)
preds = model2.predict(X_test_sm)
print("\nRefit R^2 (train):", model2.rsquared)
print("Refit Test R^2:", r2_score(y_test, preds))
print("\nUnemployment coefficient (refit):", model2.params['Unemployment'])

# Visualization: actual vs predicted poverty rate
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(y_test, preds, alpha=0.4, s=15)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, 'r--', linewidth=1)
plt.xlabel('Actual Poverty Rate (%)')
plt.ylabel('Predicted Poverty Rate (%)')
plt.title(f'Actual vs Predicted County Poverty Rate (R² = {r2_score(y_test, preds):.2f})')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved actual_vs_predicted.png")
