
import pandas as pd
import numpy as np
import os

import streamlit as st

# Correct absolute path with your username
file_path = os.path.join(os.path.dirname(__file__), "Telco_customer_churn.xlsx")
df = pd.read_excel(file_path)

df = df.loc[:, ~df.columns.duplicated()]

# Load dataset once and normalize column names
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower()


# Quick look at the data
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())
print(df.describe())


# Create a single churn column
if 'churn label' in df.columns:
    df['churn_value'] = df['churn label'].map({'Yes': 1, 'No': 0})
elif 'churn' in df.columns:
    df['churn_value'] = df['churn'].map({'Yes': 1, 'No': 0})

# Verify dataset has both churned (1) and non-churned (0) customers
print("Churn column check:")
print(df['churn_value'].unique())
print(df['churn_value'].value_counts())


# Check raw churn column distribution here
print("Churn column check:")
print(df['churn_value'].unique())
print(df['churn_value'].value_counts())

print("Churn distribution:\n", df['churn_value'].value_counts())

# Now copy the full df (with churn_value included) for revenue loss calculations later
df_original = df.copy()

# Drop duplicate columns and indices once
df_original = df_original.loc[:, ~df_original.columns.duplicated()]
df_original = df_original.loc[~df_original.index.duplicated()]

# Clean df_original to avoid duplicate column/index issues
df_original = df_original.loc[:, ~df_original.columns.duplicated()]   # drop duplicate columns
df_original = df_original.loc[~df_original.index.duplicated()]        # drop duplicate indices


if 'total charges' in df.columns:
    df['total charges'] = pd.to_numeric(df['total charges'], errors='coerce')


# Only drop rows where churn_value is missing
df = df.dropna(subset=['total charges', 'monthly charges', 'tenure months'])

print(df.columns.tolist())

import seaborn as sns
import matplotlib.pyplot as plt

# Churn vs Tenure Months
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='churn_value', y='tenure months', data=df)
plt.title('Churn vs Tenure Months')
plt.show()

plt.title('Churn vs Tenure Months')
plt.show()

# Churn vs Monthly Charges
sns.boxplot(x='churn_value', y='monthly charges', data=df)
plt.title('Churn vs Monthly Charges')
plt.show()

# churn vs contract type
sns.countplot(x='contract', hue='churn_value', data=df)
plt.title('Churn vs Contract Type')
plt.show()

# Churn vs Payment Method
sns.countplot(x='payment method', hue='churn_value', data=df)
plt.title('Churn vs Payment Method')
plt.show()

# correlation heatmap
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# distribution of numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show() 

# feature relevance tenure group (0-12, 12-24,)
df['tenure_group'] = pd.cut(df['tenure months'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])
sns.countplot(x='tenure_group', hue='churn label', data=df)
plt.title('Churn vs Tenure Group')
plt.show() 

# combine services into a single feature
service_cols = ['phone service', 'multiple lines', 'internet service', 'online security', 'online backup', 'device protection', 'tech support', 'streaming tv', 'streaming movies']
df['total_services'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
sns.countplot(x='total_services', hue='churn label', data=df)
plt.title('Churn vs Total Services')
plt.show() 

# Clean numeric columns before scaling
for col in ['tenure months', 'monthly charges', 'total charges']:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # convert strings/blanks to NaN

# Drop rows where any of these key numeric columns are missing
df = df.dropna(subset=['tenure months', 'monthly charges', 'total charges'])

# Now scale safely
from sklearn.preprocessing import StandardScaler
numeric_features = ['tenure months', 'monthly charges', 'total charges']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Drop irrelevant columns but keep churn_value
drop_cols = ['customerid','country','state','city','zip code','lat long','churn reason','churn label','churn']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Quick check
print(df.head())

# Standardize target column name
df = df.rename(columns={'churn value': 'churn_value'})

# Drop leakage columns before encoding to prevent dummy churn features
drop_cols = [
    'customerid','country','state','city','zip code','lat long',
    'churn reason','churn label','churn','churn score','cltv'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])


# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

#Translate insights/results into business language
print("Business Insights:")
print("Contract type and monthly charges are the most predictive of churn. Customers on month-to-month contracts are 3x more likely to churn.")


# Clean and convert monthly charges
df['monthly charges'] = (
    df['monthly charges']
    .astype(str)
    .str.strip()
    .str.replace('[^0-9.]', '', regex=True)
)
df['monthly charges'] = pd.to_numeric(df['monthly charges'], errors='coerce')


# Business insights / revenue loss
# Use df_original because it has the raw values before scaling/encoding

# Clean monthly charges column
df_original['monthly charges'] = (
    df_original['monthly charges']
    .astype(str)
    .str.strip()
    .str.replace('[^0-9.]', '', regex=True)
)
df_original['monthly charges'] = pd.to_numeric(df_original['monthly charges'], errors='coerce')

# Debug check for duplicates
print("Duplicate columns:", df_original.columns[df_original.columns.duplicated()])
print("Duplicate index:", df_original.index[df_original.index.duplicated()])

print("Columns:", df_original.columns.tolist())
print("Count of churn_value columns:", (df_original.columns == "churn_value").sum())

# Now safe to filter churned customers
churned_customers = df_original[df_original['churn_value'] == 1]

# Calculate churn metrics
monthly_loss = churned_customers['monthly charges'].sum()
annual_loss = monthly_loss * 12
churn_rate = churned_customers.shape[0] / df_original.shape[0]
avg_loss_per_customer = monthly_loss / churned_customers.shape[0]

# Print results in business language
print(f"\n📊 BUSINESS INSIGHTS")
print(f"Churn rate: {churn_rate:.2%}")
print(f"Estimated monthly revenue lost due to churn: R{monthly_loss:,.2f}")
print(f"Estimated annual revenue lost due to churn: R{annual_loss:,.2f}")
print(f"Average monthly revenue lost per churned customer: R{avg_loss_per_customer:,.2f}")

# Feature Engineering: Interaction Terms
# Interaction between contract type and monthly charges
# Create a month-to-month indicator manually
df['contract_month_to_month'] = ((df['contract_One year'] == 0) & (df['contract_Two year'] == 0)).astype(int)

# Interaction term: monthly charges × month-to-month contract
df['contract_monthly_interaction'] = df['monthly charges'] * df['contract_month_to_month']
print("New engineered features:\n", df[['monthly charges', 'contract_month_to_month', 'contract_monthly_interaction']].head())


# Define features and target (remove leakage)
leakage_cols = [
    'churn_value',   # target
    'churn score',   # directly related to churn
    'cltv',          # customer lifetime value (derived from churn)
    'churn',         # original churn column
    'churn label',   # original churn label
    'churn reason'   # reason for churn
]

X = df.drop(columns=[col for col in leakage_cols if col in df.columns])
y = df['churn_value'].iloc[:, 0] if isinstance(df['churn_value'], pd.DataFrame) else df['churn_value']

# Debug check
print("y type:", type(y))   # should be <class 'pandas.core.series.Series'>
print("Unique values in y:", y.unique())

print([col for col in X.columns if "churn" in col.lower()])


# Diagnostic check for leakage
print("Number of features in X:", len(X.columns))
print("Suspicious columns:", [col for col in X.columns if "churn" in col.lower()])

print("Columns in X:", X.columns.tolist()[:30])  # show first 30 columns


# Check class distribution
print("Class distribution before split:\n", y.value_counts())

# Stratified train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify both classes exist in training and test sets
print("Training class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# Train models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

print("Training class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# Guard Logistic Regression Training
if len(y_train.unique()) > 1:
    log_reg = LogisticRegression(max_iter=3000, solver='lbfgs', random_state=42)
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
else:
    print("❌ Error: Training data has only one class. Please check preprocessing or reload dataset.")


# Logistic Regression
# Logistic Regression (increase iterations to avoid warnings)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=3000, solver='lbfgs', random_state=42)
log_reg.fit(X_train, y_train)
log_pred = log_reg.predict(X_test)

# Logistic Regression Feature Importance
feature_importance = log_reg.coef_[0]
feature_names = X.columns
indices = np.argsort(np.abs(feature_importance))[::-1]

# Logistic Regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train, y_train)
log_reg_acc = log_reg.score(X_test, y_test)

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices[:min(15, len(feature_names))]],
         feature_importance[indices[:min(15, len(feature_names))]])
plt.xlabel('Coefficient (Impact on Churn)')
plt.title('Top Features Influencing Churn (Logistic Regression)')
plt.gca().invert_yaxis()
plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_importance = rf.feature_importances_
indices = np.argsort(rf_importance)[::-1]

# Random Forest
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)    

plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices[:min(15, len(feature_names))]],
         rf_importance[indices[:min(15, len(feature_names))]])
plt.xlabel('Importance')
plt.title('Top Features Influencing Churn (Random Forest)')
plt.gca().invert_yaxis()
plt.show()


# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

# After training Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_acc = gb_model.score(X_test, y_test)

# Compare accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))

# Gradient Boosting Feature Importance
gb_importance = gb.feature_importances_
indices = np.argsort(gb_importance)[::-1]

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

# Random Forest detailed metrics
print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# Gradient Boosting detailed metrics
print("\nGradient Boosting Report:")
print(classification_report(y_test, gb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_pred))


plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in indices[:min(15, len(feature_names))]],
         gb_importance[indices[:min(15, len(feature_names))]])
plt.xlabel('Importance')
plt.title('Top Features Influencing Churn (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.show()


st.title("Telco Customer Churn Analysis")

st.write("📊 Business Insights")
st.write("Churn rate:", "26.54%")
st.write("Estimated monthly revenue lost due to churn: R139,130.85")
st.write("Estimated annual revenue lost due to churn: R1,669,570.20")

st.write("### Model Performance")
st.metric("Logistic Regression Accuracy", "80.45%")
st.metric("Random Forest Accuracy", "78.39%")
st.metric("Gradient Boosting Accuracy", "78.96%")

# --- Streamlit interactive layout ---
tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Business Insights", "Model Results"])


with tab1:
    st.write("### Dataset Overview")
    # Ensure unique column names before showing in Streamlit
    df_clean = df.loc[:, ~df.columns.duplicated()]
    st.dataframe(df_clean.head())
    st.bar_chart(df_clean['churn_value'].value_counts())

with tab2:
    st.header("Business Insights")

    # Local slider for exploration
    min_tenure = st.slider("Minimum tenure months", 0, int(df["tenure months"].max()), 0)
    filtered_df = df[df["tenure months"] >= min_tenure]

    # Churn metrics
    churn_rate = filtered_df["churn_value"].mean() * 100
    monthly_loss = filtered_df[filtered_df["churn_value"] == 1]["monthly charges"].sum()
    annual_loss = monthly_loss * 12

    st.write(f"Churn Rate: {churn_rate:.2f}%")
    st.write(f"Estimated Monthly Revenue Lost: R{monthly_loss:,.2f}")
    st.write(f"Estimated Annual Revenue Lost: R{annual_loss:,.2f}")

    # Scenario testing (numeric input)
    drop_rate = st.number_input("What if churn rate drops by (%)", 0, 100, 5)
    new_loss = annual_loss * (1 - drop_rate/100)
    st.write(f"New Annual Loss if churn drops: R{new_loss:,.2f}")

    # Download button
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Insights (CSV)", csv, "business_insights.csv", "text/csv")

with tab3:
    st.write("### Model Results")
    model_choice = st.radio("Select model:", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    
    if model_choice == "Logistic Regression":
        st.metric("Accuracy", f"{log_reg_acc*100:.2f}%")
    elif model_choice == "Random Forest":
        st.metric("Accuracy", f"{rf_acc*100:.2f}%")
    else:
        st.metric("Accuracy", f"{gb_acc*100:.2f}%")








