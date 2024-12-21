"""
Module 1: Understanding the data
Description: Functions for data cleaning and statistical analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

# Load the data
kc_data = pd.read_csv("/Users/selinwork/Documents/Ironhack/Ironhack_Week_5/Project_IronKaggle/king_ country_ houses_aa.csv")

# Data formatting
kc_data.columns = kc_data.columns.str.lower().str.replace(' ', '_')

# make a copy the dataset
df = kc_data.copy()

# Understanding the data type
df.dtypes

# Check for missing values
df.isnull().sum()

# Check for duplicates and empty values
df.duplicated().sum() 
df.eq(" ").sum()  

# Ensure 'date' is in datetime format
df["date"] = pd.to_datetime(df["date"], errors='coerce')

# Create 'year_month' column as a numerical format
df['year_month'] = df['date'].dt.year * 100 + df['date'].dt.month

"""
Module 2: EDA and Data Visualization
Description: Functions for data visualization and statistical analysis.
"""

# Descriptive statistics
df['price_per_sqft'] = round(df['price'] / df['sqft_living'], 2)

# Summary statistics
target = df.pop("price")
df["price"] = target

# Correlation matrix
df.corrwith(df["price"]).sort_values(ascending=False)

# Statistical summary
round(num.describe().T, 2)

df_features = df.groupby('price_per_sqft')["grade"].mean().reset_index()

# Visualizations

"""
Module 3.1: Machine Learning
Description: Functions for data preprocessing and machine learning.
"""
# Copy my dataset to work on ML
ml_df = df.copy()

# Drop columns
ml_df = df.drop(["id","price_per_sqft"], axis=1)

# Dropped my target from the features and assigned it to y
X = ml_df.drop(["price"], axis=1)
y = ml_df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Testing the different models and deciding which one to use

# Linear Regression model and comparison with actual values
# Ridge 
# Lasso
# Decision TreeRegressor
# KNeighborsRegressor
# XGBRegressor

"""
Module 3.2: Machine Learning
"""

# XGBRegressor is the best model to use

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def root_mean_squared_error(y_true, y_pred):
	return root_mean_squared_error(y_true, y_pred, squared=False)

r2_8 = r2_score(y_test, pred_xgb)
RMSE_8 = root_mean_squared_error(y_test, pred_xgb)
MSE_8 = mean_squared_error(y_test, pred_xgb)
MAE_8 = mean_absolute_error(y_test, pred_xgb)

print("R2 = ", round(r2_8, 4))
print("RMSE = ", round(RMSE_8, 4))
print("The value of the metric MSE is ", round(MSE_8, 4))
print("MAE = ", round(MAE_8, 4))

# I want to try Scalers and GridSearchCV to improve the model

scaler = MinMaxScaler()
scaler = StandardScaler()

# Since they didn't improve the model and moved on normalization with Log Transformation

X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# It didn't give me the results I wanted, so I tried feature importance and moved on to GridSearchCV

importance = xgbr.feature_importances_
indices = np.argsort(importance)[::-1]

# GridSearchCV to find the best parameters to get a best result

# Ran the GridSearchCV and got the best parameters
grid.best_params_

# I used the best parameters and got the best model

# I found my optimum model and results.










