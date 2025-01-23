import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt

# Load the trained LSTM model
model = load_model('Stock_Price_Model.h5')

# Load the CSV containing Saudi companies' data
csv_path = 'Equites_Historical_Adjusted_Prices_Report.csv'
df = pd.read_csv(csv_path)

# Function to create sequences
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Function to fetch stock data for the past year using yfinance
def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period='1y')
    return stock_data

# Function to predict the next week's prices
def predict_next_week(symbol):
    ticker = f"{symbol}.SR"
    stock_data = fetch_stock_data(ticker)

    if stock_data.empty:
        print(f"No data fetched for {symbol}.")
        return None

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    if len(data_scaled) < 50:
        print(f"Not enough data to create a sequence for {symbol}. Minimum required: 50, available: {len(data_scaled)}")
        return None

    # Create sequences
    sequence_length = 50
    sequences = create_sequences(data_scaled, sequence_length)
    X = sequences[-1].reshape(1, sequence_length, 1)

    # Make predictions
    predictions_scaled = model.predict(X)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    start_date = stock_data.index[-1]
    return start_date, predictions

# Plotting function
def plot_predictions(start_date, predictions, symbol):
    future_dates = [start_date + pd.Timedelta(days=i+1) for i in range(len(predictions))]
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, predictions, marker='o', linestyle='-', label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Predicted Closing Price')
    plt.title(f'Predicted Stock Prices for {symbol}')
    plt.legend()
    plt.show()

# Example usage
example_symbol = "1020"
result = predict_next_week(example_symbol)
if result:
    start_date, predictions = result
    print(f"Predictions from {start_date}: {predictions}")
    plot_predictions(start_date, predictions, example_symbol)
else:
    print("Prediction failed.")
