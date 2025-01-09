import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
data_path = 'Equites_Historical_Adjusted_Prices_Report.csv'
df = pd.read_csv(data_path)

# Convert 'Date' to datetime and sort values
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date'])

# Select relevant columns for analysis
data = df[['Symbol', 'Date', 'Close', 'Volume Traded']]

# Filter for a single symbol for simplicity (e.g., Riyad Bank)
# Make sure '1010' is a valid symbol in your dataset
# If not, replace it with a valid one
symbol_data = data[data['Symbol'].astype(str).str.lower() == '1010'.lower()]

# Check if symbol_data is empty
if symbol_data.empty:
    print(f"No data found for symbol '1010'. Check if it's a valid symbol in your dataset.")
else:
    symbol_data = symbol_data[['Date', 'Close']].set_index('Date')

    # Normalize the 'Close' prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(symbol_data['Close'].values.reshape(-1, 1))

    # Create sequences for LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 50  # Use the last 50 days to predict the next day
X, y = create_sequences(data_scaled, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape the data to 3D (required for LSTM input)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Predict for the next day
last_sequence = data_scaled[-sequence_length:]
last_sequence = last_sequence.reshape((1, last_sequence.shape[0], 1))
next_day_prediction = model.predict(last_sequence)
next_day_prediction = scaler.inverse_transform(next_day_prediction)
print(f'Next Day Predicted Close Price: {next_day_prediction[0, 0]}')

# prompt: create a plot that shows all the previous closing prices and the predict price of the company being predicted 

import matplotlib.pyplot as plt

# Assuming 'symbol_data', 'y_test', 'y_pred', and 'next_day_prediction' are defined from the previous code

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(symbol_data.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(symbol_data.index[-len(y_pred):], y_pred, label='Predicted Prices')

# Plot the next day's prediction
last_date = symbol_data.index[-1]
next_date = last_date + pd.Timedelta(days=1)  # Calculate the next date
plt.scatter(next_date, next_day_prediction[0, 0], color='red', label='Next Day Prediction', marker='x', s=100)

plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Closing Prices')
plt.legend()
plt.grid(True)
plt.show()