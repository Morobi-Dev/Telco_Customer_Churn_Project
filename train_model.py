"""
train_model.py
----------------
Run this file ONCE (or whenever your data changes) to load the Telco churn
data, clean it, engineer features, and train the 3 models.

It saves everything the Streamlit app needs into a single file called
'model_artifacts.pkl'. The app then just loads that file instead of
redoing all this work on every page refresh.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading data...")
file_path = os.path.join(os.path.dirname(__file__), "Telco_customer_churn.xlsx")
df = pd.read_excel(file_path)
df = df.loc[:, ~df.columns.duplicated()]
df.columns = df.columns.str.strip().str.lower()

# Create a single churn column
if 'churn label' in df.columns:
    df['churn_value'] = df['churn label'].map({'Yes': 1, 'No': 0})
elif 'churn' in df.columns:
    df['churn_value'] = df['churn'].map({'Yes': 1, 'No': 0})

if 'total charges' in df.columns:
    df['total charges'] = pd.to_numeric(df['total charges'], errors='coerce')

df = df.dropna(subset=['total charges', 'monthly charges', 'tenure months'])

# Keep a clean, unscaled copy for the dashboard to display (real-world values)
df_original = df.copy()
df_original = df_original.loc[:, ~df_original.columns.duplicated()]
df_original = df_original.loc[~df_original.index.duplicated()]
df_original['monthly charges'] = (
    df_original['monthly charges'].astype(str).str.strip()
    .str.replace('[^0-9.]', '', regex=True)
)
df_original['monthly charges'] = pd.to_numeric(df_original['monthly charges'], errors='coerce')

# Business metrics (used on the Overview page)
churned_customers = df_original[df_original['churn_value'] == 1]
monthly_loss = churned_customers['monthly charges'].sum()
annual_loss = monthly_loss * 12
churn_rate = churned_customers.shape[0] / df_original.shape[0]
avg_loss_per_customer = monthly_loss / churned_customers.shape[0]

print(f"Churn rate: {churn_rate:.2%}")
print(f"Monthly revenue lost: R{monthly_loss:,.2f}")
print(f"Annual revenue lost: R{annual_loss:,.2f}")

# ---------------- Feature engineering for modeling ----------------
print("Preparing features...")

for col in ['tenure months', 'monthly charges', 'total charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['tenure months', 'monthly charges', 'total charges'])

numeric_features = ['tenure months', 'monthly charges', 'total charges']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

drop_cols = [
    'customerid', 'country', 'state', 'city', 'zip code', 'lat long',
    'churn reason', 'churn label', 'churn', 'churn score', 'cltv'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

df = pd.get_dummies(df, drop_first=True)
df = df.rename(columns={'churn value': 'churn_value'})

df['monthly charges'] = (
    df['monthly charges'].astype(str).str.strip()
    .str.replace('[^0-9.]', '', regex=True)
)
df['monthly charges'] = pd.to_numeric(df['monthly charges'], errors='coerce')

# Interaction feature: month-to-month contract x monthly charges
if 'contract_One year' in df.columns and 'contract_Two year' in df.columns:
    df['contract_month_to_month'] = ((df['contract_One year'] == 0) & (df['contract_Two year'] == 0)).astype(int)
    df['contract_monthly_interaction'] = df['monthly charges'] * df['contract_month_to_month']

# Remove leakage columns before training
leakage_cols = ['churn_value', 'churn score', 'cltv', 'churn', 'churn label', 'churn reason']
X = df.drop(columns=[col for col in leakage_cols if col in df.columns])
y = df['churn_value'].iloc[:, 0] if isinstance(df['churn_value'], pd.DataFrame) else df['churn_value']

feature_names = X.columns

# ---------------- Train / test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Train models ----------------
print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=3000, solver='lbfgs', class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_pred)
feature_importance = log_reg.coef_[0]

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_importance = rf_model.feature_importances_

print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
gb_importance = gb_model.feature_importances_

print("\nModel accuracy:")
print(f"  Logistic Regression: {log_reg_acc:.2%}")
print(f"  Random Forest:       {rf_acc:.2%}")
print(f"  Gradient Boosting:   {gb_acc:.2%}")

print("\nLogistic Regression report:\n", classification_report(y_test, log_pred))
print("Random Forest report:\n", classification_report(y_test, rf_pred))
print("Gradient Boosting report:\n", classification_report(y_test, gb_pred))

# ---------------- Save everything the app needs ----------------
artifacts = {
    "df_original": df_original,
    "churn_rate": churn_rate,
    "monthly_loss": monthly_loss,
    "annual_loss": annual_loss,
    "avg_loss_per_customer": avg_loss_per_customer,
    "feature_names": feature_names,
    "log_reg_acc": log_reg_acc,
    "rf_acc": rf_acc,
    "gb_acc": gb_acc,
    "feature_importance": feature_importance,
    "rf_importance": rf_importance,
    "gb_importance": gb_importance,
    "log_reg_report": classification_report(y_test, log_pred, output_dict=True),
    "rf_report": classification_report(y_test, rf_pred, output_dict=True),
    "gb_report": classification_report(y_test, gb_pred, output_dict=True),
}

out_path = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")
joblib.dump(artifacts, out_path)
print(f"\nSaved artifacts to {out_path}")
print("You can now run: streamlit run app.py")