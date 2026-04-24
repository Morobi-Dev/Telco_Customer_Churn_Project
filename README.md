# Customer Churn Analysis - Telco Dataset
Telco international is losing customers(churn),leading to significant revenue loss. 

This project aims to analyse customer data to identify key drivers of churn, predict at-risk customers, and provide data-driven recommendations to improve customer retention.

# Key Business Questions
1. Which customers are likely to churn?
2. What factors have the greatest influence on churn?
3. What is the financial impact of churn?

# Tools & Technologies
1. Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn)
2. Excel
3. VS Code

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
   - Built a Logistic Regression model to predict churn
   - Evaluated model performance using accuracy, confusion matrix, and classification report
  
# Key Insights
  - Customers on month-to-month contracts have a significantly higher churn rate
  - Customers with higher monthly charges are more likely to churn
  - Customers with low tenure (0–12 months) are at the highest risk of leaving
  - Customers with fewer subscribed services show increased churn behaviour

# Key Insights
  - Estimated monthly revenue loss due to churn: $1500.82
  - Estimated annual revenue loss: $18,009.90
  - Average monthly revenue lost per churned customer: $0.80

# Recommendations(So Far)
  - Introduce incentives for long-term contracts
  - Improve onboarding experience for new customers
  - Review pricing strategies for high-paying customers
  - Develop targeted retention campaigns for high-risk groups

# Project Status
Work in Progress, Ongoing improvements include model optimisation and dashboard development.


