import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
data = pd.read_csv('dataset.txt', delimiter=',')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Check for missing values
data = data.dropna()

# Create train-test split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the ARIMA model using auto_arima
model = auto_arima(train, seasonal=False, stepwise=True, trace=True)

# Forecast the future values
forecast_steps = len(test)
forecast = model.predict(n_periods=forecast_steps)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse}')

# Plotting the actual vs forecasted values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label="Train Data")
plt.plot(test.index, test, label="Test Data")
plt.plot(test.index, forecast, label="Forecast", color='red')

# Forecasting future values beyond the test data
future_forecast = model.predict(n_periods=11)

# Plotting the future forecast
plt.plot(pd.date_range(data.index[-1], periods=12, freq='ME')[1:], future_forecast, label="Future Forecast", color='green')

# Add labels and title
plt.title('ARIMA Model Forecasting (auto_arima)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.savefig('ARIMA_Forecast.jpg')
