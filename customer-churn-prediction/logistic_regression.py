"""
Customer Churn Prediction — Logistic Regression
Predicts telecom customer churn and identifies the strongest churn drivers.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("churn_clean.csv")

# Handle missing values
df['InternetService'] = df['InternetService'].astype('category')
if 'None' not in df['InternetService'].cat.categories:
    df['InternetService'] = df['InternetService'].cat.add_categories('None')
df['InternetService'] = df['InternetService'].fillna('None')

# Select relevant columns, drop identifiers/irrelevant fields
columns_to_drop = [
    'CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County',
    'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 'Job', 'Email',
    'Contacts', 'Yearly_equip_failure', 'Marital', 'Gender'
]
df = df.drop(columns=columns_to_drop, errors='ignore')
relevant_columns = [
    'Churn', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year',
    'Outage_sec_perweek', 'Age', 'Income', 'Contract', 'InternetService',
    'PaymentMethod'
]
df = df[relevant_columns]

# Encode categoricals, treat outliers
categorical_cols = ['Contract', 'InternetService', 'PaymentMethod']
for col in categorical_cols:
    df[col] = df[col].astype('category')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

numerical_cols = ['Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Outage_sec_perweek', 'Age', 'Income']
for col in numerical_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df_encoded = df_encoded.loc[df.index]
df_encoded['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df_encoded.drop(columns=['Churn']).astype(float)
y = df_encoded['Churn'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial model (statsmodels for inference, sklearn for accuracy)
X_train_sm = sm.add_constant(X_train.values)
initial_result = sm.Logit(y_train.values, X_train_sm).fit(disp=0)

initial_model_sk = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
initial_model_sk.fit(X_train, y_train)
initial_accuracy = accuracy_score(y_test, initial_model_sk.predict(X_test))
print(f"Initial model accuracy: {initial_accuracy:.4f}")
print(f"Initial model pseudo R-squared: {initial_result.prsquared:.4f}")

# Reduced model via Recursive Feature Elimination (top 10 features)
rfe = RFE(estimator=LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'), n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_].tolist()
print(f"\nSelected features (RFE): {selected_features}")

reduced_model_sk = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear', class_weight='balanced')
reduced_model_sk.fit(X_train[selected_features], y_train)
y_pred_reduced = reduced_model_sk.predict(X_test[selected_features])
reduced_accuracy = accuracy_score(y_test, y_pred_reduced)

print(f"\nReduced model accuracy: {reduced_accuracy:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_reduced))
print("\nClassification report:\n", classification_report(y_test, y_pred_reduced))

coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': reduced_model_sk.coef_[0]})
print("\nFeature coefficients (reduced model):\n", coef_df.sort_values('Coefficient').round(4))
