"""
app.py
-------
The Streamlit dashboard. This file does NOT train any models — it just
loads the results that train_model.py already prepared and saved to
'model_artifacts.pkl'. This makes the app load fast.

If you see a "model_artifacts.pkl not found" error, run:
    python train_model.py
first, then re-run this app.
"""

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn — Enterprise Dashboard", page_icon="📊", layout="wide")

# ---------------- Load pre-trained artifacts ----------------
artifacts_path = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")

if not os.path.exists(artifacts_path):
    st.error(
        "Model artifacts not found. Please run `python train_model.py` once "
        "in this folder before launching the dashboard."
    )
    st.stop()

artifacts = joblib.load(artifacts_path)

df_original = artifacts["df_original"]
global_churn_rate = artifacts["churn_rate"] * 100
global_monthly_loss = artifacts["monthly_loss"]
global_annual_loss = artifacts["annual_loss"]
feature_names = artifacts["feature_names"]
log_acc = artifacts["log_reg_acc"] * 100
rf_acc_val = artifacts["rf_acc"] * 100
gb_acc_val = artifacts["gb_acc"] * 100
feature_importance = artifacts["feature_importance"]
rf_importance = artifacts["rf_importance"]
gb_importance = artifacts["gb_importance"]

churned_customers = df_original[df_original['churn_value'] == 1]

# ---------------- Styling ----------------
st.markdown("""
<style>
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

[data-testid="stSidebar"] {
  background-color: var(--sidebar-bg);
}

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

.insight-box {
  background: linear-gradient(180deg, rgba(17,33,41,0.45), rgba(11,23,32,0.35));
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(6,182,212,0.06);
}

.footer {
  color: var(--muted);
  font-size:12px;
  padding: 12px 0 30px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
df_display = df_original.copy()
if not all(col.islower() for col in df_display.columns):
    df_display.columns = [c.lower() for c in df_display.columns]

with st.sidebar:
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
        except Exception:
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

# ---------------- KPI cards ----------------
col1, col2, col3, col4 = st.columns([1.6, 1.2, 1.2, 1.2], gap='large')

with col1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Churn Rate</div>
        <div class='kpi-value'>{global_churn_rate:.2f}%</div>
        <div class='kpi-trend'>↓ 3.2% vs last period</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Monthly Revenue Lost</div>
        <div class='kpi-value'>R{global_monthly_loss:,.0f}</div>
        <div class='kpi-trend' style='color:#F59E0B'>High priority</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Annualised Loss</div>
        <div class='kpi-value'>R{global_annual_loss:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>Top Model (Logistic Reg)</div>
        <div class='kpi-value'>{log_acc:.2f}%</div>
        <div class='kpi-trend'>RF {rf_acc_val:.2f}% • GB {gb_acc_val:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- Pages ----------------
if page == 'Overview':
    st.header("Executive Summary")
    st.markdown("<div class='insight-box'><strong>Key Point:</strong> Contract type and monthly charges are the strongest predictors of churn. Focus retention offers on month-to-month customers.</div>", unsafe_allow_html=True)
    st.metric("Overall Churn Rate", f"{global_churn_rate:.2f}%")

    st.subheader("Dataset Snapshot")
    df_clean_main = df_display.loc[:, ~df_display.columns.duplicated()]
    st.dataframe(df_clean_main.head(10))

    st.subheader("Churn Distribution")
    try:
        dist_col1, dist_col2 = st.columns([2, 3])
        with dist_col1:
            st.bar_chart(df_clean_main['churn_value'].value_counts())
        with dist_col2:
            try:
                counts = df_clean_main['churn_value'].value_counts().reindex([0, 1], fill_value=0)
                labels = ['No', 'Yes']
                colors = ['#14B8A6', '#06B6D4']
                fig, ax = plt.subplots(figsize=(6, 3))
                wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': 'white'})
                centre_circle = plt.Circle((0, 0), 0.65, fc='#11212D')
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
    df_ins = df_display.loc[:, ~df_display.columns.duplicated()]
    df_ins = df_ins[(df_ins['tenure months'] >= tenure_filter[0]) & (df_ins['tenure months'] <= tenure_filter[1]) & (df_ins['monthly charges'] <= monthly_filter)]

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Revenue at Risk by Contract Type")
        try:
            if 'contract' in df_ins.columns:
                contract_loss = df_ins[df_ins['churn_value'] == 1].groupby('contract')['monthly charges'].sum()
            else:
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
            coeffs = list(zip(feature_names, feature_importance))
            coeffs_df = pd.DataFrame(coeffs, columns=['feature', 'coef']).sort_values('coef', key=lambda x: x.abs(), ascending=False).head(12)
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

# ---------------- Footer ----------------
st.markdown("<div class='footer'>Developed by Morobi Mofokeng — Data Science | Business Intelligence | Machine Learning — © 2026 All Rights Reserved</div>", unsafe_allow_html=True)