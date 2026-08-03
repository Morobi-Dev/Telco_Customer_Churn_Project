# Customer Churn Analysis - Telco Dataset

## Live Demo
Try the interactive app here: [Streamlit Cloud Link](https://telcocustomerchurnproject-qws5sjxy2g7pvvr9cossr3.streamlit.app/)

## Project Overview
Telco international is losing customers(churn),leading to significant revenue loss. 
This project aims to analyse customer data to identify key drivers of churn,predict at-risk customers,
and provide actionable recommendations to improve customer retention strategies.

By combining predictive modelling with business insights. the project demonstrates how data science
can directly support decision-making and reduce financial risk.

# Key Business Questions
1. Which customers are likely to churn?
2. What factors have the greatest influence on churn?
3. What is the financial impact of churn?
4. How can targeted strategies reduce churn and protect revenue?

# Tools & Technologies
1. Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn)
2. Excel (Initial data exploration and validation)
3. VS Code (Development environment)

# Poject Approach
1. Data Cleaning & Preparation
   - Handled missing values and corrected data types
   - Converted categorical variables into numerical format
   - Standardised column names for consistency
  
2. Exploratory Data Analysis (EDA)
   - Analysed churn distribution across key variables
   - Identified trends in contract type, tenure, and pricing
   - Visualised patterns using plots and charts

3. Feature Engineering
   - Created tenure groups to analyse customer lifecycle stages
   - Calculated total services per customer

4. Predictive Modelling
   - Built a Logistic Regression, Random Forest, and Gradient Boosting model to predict churn
   - Evaluated performance using accuracy, confusion matrix, and classification report
  
# Key Insights
  - Customers on month-to-month contracts have a significantly higher churn rate
  - Customers with higher monthly charges are more likely to churn
  - Customers with low tenure (0–12 months) are at the highest risk of leaving
  - Customers with fewer subscribed services show increased churn behaviour
  - Certain payment methods show higher churn rates
  - Machine learning models consistently identified Contract Type, Monthly Charges and Tenure as the strongest predictors.

# Business Impact
  - Estimated monthly revenue loss due to churn: R1500.82
  - Estimated annual revenue loss: R18,009.90
  - Average monthly revenue lost per churned customer: R0.80
  - Scenario testing shows that reducing churn by just 5% could save R83,000 annually

# Recommendations
  - Introduce incentives for long-term contracts
  - Improve onboarding experience for new customers
  - Review pricing strategies for high-paying customers
  - Develop targeted retention campaigns for high-risk groups
  - Prioritise retention for customers with high CLTV.
  - Develop predictive dashboards highlighting high churn score customers.

# Visualisations
   ![Churn Distribution](Visuals/churn_distribution.png)  
   ![Feature Importance](Visuals/feature_importance.png)  
   ![Revenue Loss](Visuals/revenue_loss.png)  


# This analysis demonstrates that customer churn is not random. It is strongly influenced by customer tenure, contract type,
  monthly charges, service adoption, and payment behaviour. By focusing retention efforts on these high-risk customer segments,
  the business can reduce churn, improve customer lifetime value, and protect recurring revenue.


# END!!!










   


   



