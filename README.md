# Credit Risk Modeling
Project: Credit Risk Modeling - Prediction of defaulter risk

Aim of the project: Calculate credit score based on the applicant's loan requirement, income, and past payment history details etc., and predicting the credit default risk of applicant.

Dataset: Customer details, Loan details and Credit Bureau Data of customer

## Data Cleaning and Exploratory Data Analysis
- Handling of Missing values
- Handling of Duplicate values
- Univariate analysis of Numerical Columns and Categorical columns
- Handling Outliers using box plots
- Bivariate analysis with bar plots and KDE plots
- Correlation observation of features

<img width="1261" height="908" alt="image" src="https://github.com/user-attachments/assets/8ee586cc-e0f4-4271-80f1-f7d92741d516" />

<img width="1264" height="917" alt="image" src="https://github.com/user-attachments/assets/526bb9ba-16db-4ce8-8314-4516d2485b83" />

## Feature Engineering and Feature Selection

Adding new columns by calculating ratios of existing columns (eg., delinquent_months_ratio , credit_utilization_per_income)

- Feature Selection with the help of relevant variance_inflation_factor, and correlation matrix
- Data preprocessing using MinMax scaling and encoding

<img width="322" height="526" alt="image" src="https://github.com/user-attachments/assets/63cb6ec6-fe45-44ad-80fa-1848f3bb58c1" />

<img width="945" height="924" alt="image" src="https://github.com/user-attachments/assets/15c03fcc-d69f-47c7-b0ee-1b63eb35f717" />

## Model Training and Fine Tuning

- Trained with Logistic Regression optimizing with Bayesian Optimization
- Also trained with XGBoostClassifier optimizing with GridSearchCV Hyperparameter tuning

<img width="874" height="389" alt="image" src="https://github.com/user-attachments/assets/c01a294b-0a9a-499b-8e04-47f68a76042f" />

## Model Evaluation and Interpretation
### Shap
SHAP (SHapley Additive exPlanations) is a powerful and widely used framework for interpreting predictions of machine learning models. It explains the output of a model by computing the contribution of each feature to the prediction.

<img width="821" height="701" alt="image" src="https://github.com/user-attachments/assets/1d712fc2-4df7-4320-8d12-7eea2f0a181b" />

<img width="1365" height="185" alt="image" src="https://github.com/user-attachments/assets/67e46f39-70ef-43b7-bdf8-3cd2c15b379a" />

## Model deploy and Prediction using Streamlit App

<img width="1369" height="935" alt="image" src="https://github.com/user-attachments/assets/1a0360e0-ff4a-469b-8e9a-19f54d39f45e" />

