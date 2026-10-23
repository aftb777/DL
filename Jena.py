# Practical 16: Temperature Forecasting Using LSTM
# Aim: Predict temperature using past temperature readings
# and evaluate the model using MAE and RMSE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load dataset
df = pd.read_csv("jena_climate_2009_2016.csv")

# Select temperature and keep one reading per hour
temperature = df["T (degC)"].values[::6].reshape(-1, 1)

# 2. Split chronologically: 80% training, 20% testing
split = int(len(temperature) * 0.8)

# 3. Normalize using training data only
scaler = MinMaxScaler()
scaler.fit(temperature[:split])

scaled_data = scaler.transform(temperature)

# 4. Create sequences
# Previous 24 hourly readings predict the next hour
lookback = 24

def create_sequences(data, start, end, lookback):
    X, y = [], []

    for i in range(max(start, lookback), end):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(
    scaled_data, lookback, split, lookback
)

X_test, y_test = create_sequences(
    scaled_data, split, len(scaled_data), lookback
)

# LSTM input shape: (samples, time steps, features)
X_train = X_train.reshape(-1, lookback, 1)
X_test = X_test.reshape(-1, lookback, 1)

# 5. Build the LSTM model
model = Sequential([
    LSTM(50, input_shape=(lookback, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# 6. Train the model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    shuffle=False
)

# 7. Predict temperature
predictions = model.predict(X_test, verbose=0)

# Convert normalized values back to Celsius
actual_temp = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

predicted_temp = scaler.inverse_transform(predictions)

# 8. Evaluate the model
mae = mean_absolute_error(actual_temp, predicted_temp)

rmse = np.sqrt(
    mean_squared_error(actual_temp, predicted_temp)
)

print("Mean Absolute Error (MAE):", round(mae, 3), "°C")
print("Root Mean Squared Error (RMSE):", round(rmse, 3), "°C")

# 9. Plot actual vs predicted temperature
plt.figure(figsize=(10, 5))

plt.plot(actual_temp[:200], label="Actual Temperature")
plt.plot(predicted_temp[:200], label="Predicted Temperature")

plt.title("LSTM Temperature Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()