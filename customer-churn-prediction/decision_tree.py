"""
Customer Churn Prediction — Decision Tree
Classifies customers into high-risk / low-risk churn groups.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("churn_clean.csv")
selected_df = df[['Contract', 'Tenure', 'MonthlyCharge', 'Churn']].copy()

valid_contracts = ['Month-to-month', 'One year', 'Two Year']
selected_df = selected_df[selected_df['Contract'].isin(valid_contracts)]

def cap_outliers(series):
    lower, upper = np.percentile(series, 1), np.percentile(series, 99)
    return np.clip(series, lower, upper)

selected_df['Tenure'] = cap_outliers(selected_df['Tenure'])
selected_df['MonthlyCharge'] = cap_outliers(selected_df['MonthlyCharge'])

contract_encoded = pd.get_dummies(selected_df['Contract'], prefix='Contract')
selected_df = pd.concat([selected_df.drop('Contract', axis=1), contract_encoded], axis=1)
selected_df['Churn'] = selected_df['Churn'].map({'No': 0, 'Yes': 1})

X = selected_df.drop('Churn', axis=1)
y = selected_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
y_prob = dt_model.predict_proba(X_test)[:, 1]
mse = np.mean((y_prob - y_test) ** 2)

print(f"Accuracy: {accuracy:.4f}")
print(f"MSE: {mse:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

importances = pd.DataFrame({'Feature': X.columns, 'Importance': dt_model.feature_importances_})
print("\nFeature importances:\n", importances.sort_values('Importance', ascending=False))
