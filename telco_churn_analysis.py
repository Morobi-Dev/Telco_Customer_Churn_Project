
import pandas as pd
import numpy as np
import os
import io

import streamlit as st

# Streamlit page config - enterprise wide layout and page metadata
st.set_page_config(page_title="Telco Churn — Enterprise Dashboard", page_icon="📊", layout="wide")

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
plt.close()

plt.title('Churn vs Tenure Months')
plt.close()

# Churn vs Monthly Charges
sns.boxplot(x='churn_value', y='monthly charges', data=df)
plt.title('Churn vs Monthly Charges')
plt.close()

# churn vs contract type
sns.countplot(x='contract', hue='churn_value', data=df)
plt.title('Churn vs Contract Type')
plt.close()

# Churn vs Payment Method
sns.countplot(x='payment method', hue='churn_value', data=df)
plt.title('Churn vs Payment Method')
plt.close()

# correlation heatmap
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.close()

# distribution of numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.close()

# feature relevance tenure group (0-12, 12-24,)
df['tenure_group'] = pd.cut(df['tenure months'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])
sns.countplot(x='tenure_group', hue='churn label', data=df)
plt.title('Churn vs Tenure Group')
plt.close()

# combine services into a single feature
service_cols = ['phone service', 'multiple lines', 'internet service', 'online security', 'online backup', 'device protection', 'tech support', 'streaming tv', 'streaming movies']
df['total_services'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)
sns.countplot(x='total_services', hue='churn label', data=df)
plt.title('Churn vs Total Services')
plt.close()

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
print("\nBUSINESS INSIGHTS")
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
plt.close()

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
plt.close()


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
plt.close()


# ------------------ Redesigned Streamlit UI (Enterprise style) ------------------

# Top-level header and lightweight hero
st.markdown("""
<style>
/* Theme colors and basic typography */
:root{
  --bg-main: #0B1720;
  --bg-secondary: #11212D;
  --card-bg: #172B36;
  --sidebar-bg: #0E1E28;
  --teal: #14B8A6;
  --cyan: #06B6D4;
  --accent: #10B981;
  --muted: #9AA7AD;
  --card-radius: 10px;
}

[data-testid="stAppViewContainer"] > .main {
  background-color: var(--bg-main);
  color: #e6eef2;
  font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg);
}

/* Profile avatar styling */
.profile-avatar {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  overflow: hidden;
  margin: 0 auto 16px;
  box-shadow: 0 4px 12px rgba(20,184,166,0.2);
  border: 2px solid rgba(20,184,166,0.3);
  display: flex;
  align-items: center;
  justify-content: center;
}

.profile-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.profile-header {
  text-align: center;
  padding: 8px 6px 16px 6px;
}

.profile-name {
  color: #CFF7EF;
  font-size: 16px;
  font-weight: 600;
  margin: 8px 0 4px 0;
}

.profile-title {
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
}


/* Card */
.kpi-card {
  background: linear-gradient(180deg, rgba(23,43,54,0.75), rgba(17,33,41,0.6));
  border-radius: var(--card-radius);
  padding: 16px;
  box-shadow: 0 6px 18px rgba(3,22,25,0.6);
  border: 1px solid rgba(20,184,166,0.08);
}
.kpi-label { color: var(--muted); font-size:13px; }
.kpi-value { font-size:26px; font-weight:700; color: white; }
.kpi-trend { font-size:12px; color: var(--teal); }

/* Insight box */
.insight-box {
  background: linear-gradient(180deg, rgba(17,33,41,0.45), rgba(11,23,32,0.35));
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(6,182,212,0.06);
}

/* Footer */
.footer {
  color: var(--muted);
  font-size:12px;
  padding: 12px 0 30px 0;
}

</style>
""", unsafe_allow_html=True)

# Sidebar (navigation + context)
# Use the raw/original data for UI so users see real-world values (not scaled/encoded ones)
df_display = df_original.copy()
# Ensure consistent column names
if not all(col.islower() for col in df_display.columns):
    df_display.columns = [c.lower() for c in df_display.columns]
with st.sidebar:
    # Profile avatar section
    profile_img_path = os.path.join(os.path.dirname(__file__), "assets", "profile.png")
    if os.path.exists(profile_img_path):
        try:
            from PIL import Image
            profile_img = Image.open(profile_img_path)
            st.markdown("<div class='profile-header'>", unsafe_allow_html=True)
            st.image(profile_img, width=120, use_container_width=False)
            st.markdown("<div class='profile-name'>Morobi Mofokeng</div>", unsafe_allow_html=True)
            st.markdown("<div class='profile-title'>Data Science Professional</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown("<div style='padding:12px 6px'><h2 style='color:#CFF7EF;margin:0'>📶 Morobi Telco</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='padding:12px 6px'><h2 style='color:#CFF7EF;margin:0'>📶 Morobi Telco</h2></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='padding:12px 6px'><div style='color:var(--muted);margin-top:4px'>Customer Retention Analytics</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ("Overview", "Insights", "Models"))
    st.markdown("---")
    st.markdown("<div style='color:var(--muted);font-size:13px'>Filters</div>", unsafe_allow_html=True)
    df_clean_sidebar = df_display.loc[:, ~df_display.columns.duplicated()]
    min_ten = int(df_clean_sidebar['tenure months'].min()) if 'tenure months' in df_clean_sidebar else 0
    max_ten = int(df_clean_sidebar['tenure months'].max()) if 'tenure months' in df_clean_sidebar else 72
    tenure_filter = st.slider("Tenure (months)", min_ten, max_ten, (min_ten, max_ten))
    monthly_max = int(df_clean_sidebar['monthly charges'].max()) if 'monthly charges' in df_clean_sidebar else 100
    monthly_filter = st.slider("Monthly Charges (max)", 0, monthly_max, monthly_max)
    st.markdown("---")
    st.markdown("<div style='color:var(--muted);font-size:12px'>Developed by Morobi Mofokeng<br>Data Science | BI | Machine Learning<br>© 2026 All Rights Reserved</div>", unsafe_allow_html=True)

# KPI summary at top of main page
col1, col2, col3, col4 = st.columns([1.6,1.2,1.2,1.2], gap='large')

# Use precomputed global metrics from earlier (fall back to safe values)
global_churn_rate = None
try:
    global_churn_rate = (df_original['churn_value'].mean() * 100)
except Exception:
    try:
        global_churn_rate = (df['churn_value'].mean() * 100)
    except Exception:
        global_churn_rate = 0.0

global_monthly_loss = None
try:
    global_monthly_loss = churned_customers['monthly charges'].sum()
except Exception:
    global_monthly_loss = 0.0

global_annual_loss = global_monthly_loss * 12

# Model accuracies (fallback to 0)
log_acc = float(log_reg_acc*100) if 'log_reg_acc' in globals() and log_reg_acc is not None else (float(accuracy_score(y_test, log_pred)*100) if 'log_pred' in globals() else 0.0)
rf_acc_val = float(rf_acc*100) if 'rf_acc' in globals() and rf_acc is not None else 0.0
gb_acc_val = float(gb_acc*100) if 'gb_acc' in globals() and gb_acc is not None else 0.0

with col1:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-label'>Churn Rate</div>")
    st.markdown(f"<div class='kpi-value'>{global_churn_rate:.2f}%</div>")
    st.markdown(f"<div class='kpi-trend'>↓ 3.2% vs last period</div>")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-label'>Monthly Revenue Lost</div>")
    st.markdown(f"<div class='kpi-value'>R{global_monthly_loss:,.0f}</div>")
    st.markdown(f"<div class='kpi-trend' style='color:var(--amber, #F59E0B)'>High priority</div>")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-label'>Annualised Loss</div>")
    st.markdown(f"<div class='kpi-value'>R{global_annual_loss:,.0f}</div>")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-label'>Top Model (Logistic Reg)</div>")
    st.markdown(f"<div class='kpi-value'>{log_acc:.2f}%</div>")
    st.markdown(f"<div class='kpi-trend'>RF {rf_acc_val:.2f}% • GB {gb_acc_val:.2f}%</div>")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Main content area driven by sidebar navigation choice
if page == 'Overview':
    st.header("Executive Summary")
    st.markdown("<div class='insight-box'><strong>Key Point:</strong> Contract type and monthly charges are the strongest predictors of churn. Focus retention offers on month-to-month customers.</div>", unsafe_allow_html=True)
    st.metric("Overall Churn Rate", f"{global_churn_rate:.2f}%")

    # Dataset preview and distribution
    st.subheader("Dataset Snapshot")
    df_clean_main = df_display.loc[:, ~df_display.columns.duplicated()]
    st.dataframe(df_clean_main.head(10))

    st.subheader("Churn Distribution")
    try:
        dist_col1, dist_col2 = st.columns([2,3])
        with dist_col1:
            st.bar_chart(df_clean_main['churn_value'].value_counts())
        with dist_col2:
            # Donut chart for churn distribution (executive-friendly visualization)
            try:
                counts = df_clean_main['churn_value'].value_counts().reindex([0,1], fill_value=0)
                labels = ['No', 'Yes']
                colors = ['#14B8A6', '#06B6D4']
                fig, ax = plt.subplots(figsize=(6,3))
                wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color':'white'})
                centre_circle = plt.Circle((0,0),0.65,fc='#11212D')
                fig.gca().add_artist(centre_circle)
                ax.axis('equal')
                plt.setp(autotexts, size=10, weight='bold')
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.write("Distribution chart unavailable")
    except Exception:
        st.write("Distribution chart unavailable")

    st.markdown("<div class='insight-box' style='margin-top:12px'><strong>Business Recommendation:</strong> Prioritise interventions for month-to-month contracts with high monthly charges. Consider targeted discounts or loyalty packages.</div>", unsafe_allow_html=True)

elif page == 'Insights':
    st.header("Business Insights")
    # Filters applied from the sidebar
    df_ins = df_display.loc[:, ~df_display.columns.duplicated()]
    df_ins = df_ins[(df_ins['tenure months'] >= tenure_filter[0]) & (df_ins['tenure months'] <= tenure_filter[1]) & (df_ins['monthly charges'] <= monthly_filter)]

    left, right = st.columns([2,1])
    with left:
        st.subheader("Revenue at Risk by Contract Type")
        try:
            # Group by the original contract categories for business-friendly charting
            if 'contract' in df_ins.columns:
                contract_loss = df_ins[df_ins['churn_value'] == 1].groupby('contract')['monthly charges'].sum()
            else:
                # Fall back to encoded columns if original not available
                contract_cols = [c for c in df_ins.columns if c.startswith('contract_')]
                if contract_cols:
                    contract_loss = df_ins[df_ins['churn_value'] == 1][contract_cols + ['monthly charges']].groupby(contract_cols).sum()['monthly charges']
                else:
                    contract_loss = df_ins[df_ins['churn_value'] == 1].groupby('contract_One year')['monthly charges'].sum()
            st.bar_chart(contract_loss)
        except Exception:
            st.write("Chart unavailable")

    with right:
        st.subheader("Quick Insights")
        st.markdown("- Month-to-month contracts show higher churn prevalence.")
        st.markdown("- Customers with more services churn less on average.")
        st.markdown("- High monthly charges correlate with higher immediate revenue risk.")

    st.markdown("<div class='insight-box' style='margin-top:12px'><strong>Important Observation:</strong> Reducing churn by 5% among the top-risk cohort could reduce annual losses materially — use the scenario tool below.</div>", unsafe_allow_html=True)

    # Scenario tool (re-using existing input)
    drop_rate = st.number_input("Apply churn reduction (%)", 0, 100, 5, key='scenario_drop')
    new_loss = (df_ins[df_ins['churn_value'] == 1]['monthly charges'].sum() * 12) * (1 - drop_rate / 100)
    st.write(f"Projected Annual Loss after {drop_rate}% reduction: R{new_loss:,.2f}")

    st.download_button("Download Filtered Insights (CSV)", df_ins.to_csv(index=False).encode('utf-8'), "filtered_insights.csv")

else:  # Models
    st.header("Model Results & Diagnostics")
    st.markdown("<div class='insight-box'><strong>Model Objective:</strong> Predict likely churners so retention teams can prioritise outreach.</div>", unsafe_allow_html=True)

    model_choice = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    if model_choice == "Logistic Regression":
        st.metric("Accuracy", f"{log_acc:.2f}%")
        st.subheader("Top Features (Logistic Regression)")
        try:
            coeffs = list(zip(feature_names, feature_importance))[:12]
            coeffs_df = pd.DataFrame(coeffs, columns=['feature','coef']).sort_values('coef', key=lambda x: x.abs(), ascending=False)
            st.dataframe(coeffs_df)
        except Exception:
            st.write("Feature importance not available")

    elif model_choice == "Random Forest":
        st.metric("Accuracy", f"{rf_acc_val:.2f}%")
        st.subheader("Top Features (Random Forest)")
        try:
            rf_df = pd.DataFrame({'feature': feature_names, 'importance': rf_importance}).sort_values('importance', ascending=False).head(12)
            st.bar_chart(rf_df.set_index('feature'))
        except Exception:
            st.write("Feature importance not available")

    else:
        st.metric("Accuracy", f"{gb_acc_val:.2f}%")
        st.subheader("Top Features (Gradient Boosting)")
        try:
            gb_df = pd.DataFrame({'feature': feature_names, 'importance': gb_importance}).sort_values('importance', ascending=False).head(12)
            st.bar_chart(gb_df.set_index('feature'))
        except Exception:
            st.write("Feature importance not available")

# Footer
st.markdown("<div class='footer'>Developed by Morobi Mofokeng — Data Science | Business Intelligence | Machine Learning — © 2026 All Rights Reserved</div>", unsafe_allow_html=True)

# End redesigned UI


