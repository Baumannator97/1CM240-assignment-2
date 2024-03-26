# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:37:07 2024

@author: jessba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.stats import kstest
from pmdarima.arima import auto_arima



#import data

data = pd.read_csv(r"C:\Users\jessba\Documents\1CM240\Assignment 2\Data Assignment 2.csv", sep=';', index_col=0)
data.index = pd.to_datetime(data.index, dayfirst=True)

#Choose which store to visualize
Store_1_demand = data['Store 1']

'''
Store_1_demand.plot(figsize=(12, 8))


#Visualization of demand over time

data.plot(figsize=(12, 8))
plt.title('Demand Over Time for Each Store')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend(title='Store')
plt.show()


# Check for seasonality using seasonal decomposition
result = seasonal_decompose(Store_1_demand, model='additive')  # Specify the period as 365 for daily data

# Visualization of decomposition components
plt.figure(figsize=(12, 8))

# Original Data
plt.subplot(4, 1, 1)
plt.plot(Store_1_demand, label='Original')
plt.title('Original Data')
plt.legend()

# Trend Component
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.title('Trend Component')
plt.legend()

# Seasonal Component
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.title('Seasonal Component')
plt.legend()

# Residual Component
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.title('Residual Component')
plt.legend()

plt.tight_layout()
plt.show()
plt.show()

#Stationary check adfuller test
# Define a significance level (commonly 0.05)
alpha = 0.05

# Loop over each store
for store in data.columns:
    # Perform Augmented Dickey-Fuller test
    result = adfuller(data[store])
    
    # Extract test statistic and p-value
    test_statistic = result[0]
    p_value = result[1]
    
    # Print results
    print(f"Store {store}:")
    print(f"Test Statistic: {test_statistic}")
    print(f"P-value: {p_value}")
    
    # Interpret results
    if p_value < alpha:
        print("Reject the null hypothesis - Data is stationary")
    else:
        print("Fail to reject the null hypothesis - Data is non-stationary")
    print()
'''
    
# Split data into training and testing sets
data = data['Store 1']
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Fit ARIMA model using auto_arima
model = auto_arima(train_data, seasonal=False, trace=True)
model.summary()

# Make predictions
forecast, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)

# Calculate RMSE
rmse = mean_squared_error(test_data, forecast, squared=False)
print("RMSE:", rmse)

# Plot the forecast and actual demand
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test')
plt.plot(test_data.index, forecast, label='Forecast')
plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Demand Forecast with ARIMA')
plt.legend()
plt.show()