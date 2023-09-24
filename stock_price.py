import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load your stock price dataset
data = pd.read_excel('stock_price.xlsx')
prices = data['Close*'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# Define a function to create input sequences and labels
def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = data[i+sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define the sequence length and split data into training and testing sets
sequence_length = 10
X, y = create_sequences(prices_scaled, sequence_length)
split = int(0.8 * len(X))
X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with dropout for regularization
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Calculate root mean squared error (RMSE)
rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Prices', color='blue')
plt.plot(y_pred_inv, label='Predicted Prices', color='red')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Predict future prices
def predict_future_prices(model, starting_data, num_days):
    forecasted_prices = []
    current_data = starting_data

    for _ in range(num_days):
        input_data = current_data.reshape(1, sequence_length, 1)
        next_day_price = model.predict(input_data)[0][0]
        forecasted_prices.append(next_day_price)

        # Update current_data for the next iteration
        current_data = np.roll(current_data, shift=-1)
        current_data[-1] = [next_day_price]

    return forecasted_prices

# Example usage to predict the next 5 days
starting_data = X_test[-1].reshape(sequence_length, 1)
forecasted_prices = predict_future_prices(model, starting_data, num_days=5)
print("Forecasted Prices for the Next 5 Days:")
print(forecasted_prices)
