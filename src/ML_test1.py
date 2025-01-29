import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'Equites_Historical_Adjusted_Prices_Report.csv'
df = pd.read_csv(data_path)

# List unique company symbols in the dataset
available_symbols = df['Symbol'].unique()
print("Available company symbols in your dataset:")
print(available_symbols)

# Load the trained model
model = load_model('stock_price_model.h5')

# Load scalers
scalers = {}

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to predict for a company symbol
def predict_stock(symbol, sequence_length=50):
    if symbol not in available_symbols:
        print("Symbol not found. Please try again.")
        return

    # Prepare data for the selected symbol
    symbol_data = df[df['Symbol'] == symbol][['Date', 'Close']].set_index('Date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(symbol_data['Close'].values.reshape(-1, 1))
    scalers[symbol] = scaler
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Predict the next 7 days
    last_sequence = data_scaled[-sequence_length:]
    next_7_days_predictions = []
    
    for _ in range(7):
        last_sequence = last_sequence.reshape((1, sequence_length, 1))
        next_day_prediction = model.predict(last_sequence)
        next_7_days_predictions.append(scaler.inverse_transform(next_day_prediction)[0, 0])
        last_sequence = np.append(last_sequence[0][1:], next_day_prediction, axis=0)

    # Display the results
    print(f"Next 7 days predictions for symbol {symbol}:")
    for i, prediction in enumerate(next_7_days_predictions):
        print(f"Day {i + 1}: {prediction:.2f}")

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(symbol_data.index[-len(y):], scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Prices')
    plt.plot(symbol_data.index[-len(y):], scaler.inverse_transform(model.predict(X).reshape(-1, 1)), label='Predicted Prices')
    future_dates = [symbol_data.index[-1] + pd.Timedelta(days=i+1) for i in range(7)]
    plt.plot(future_dates, next_7_days_predictions, label='Next 7 Days Predictions', marker='o', linestyle='--')
    print("does this work?")
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'Actual vs. Predicted Closing Prices with Next 7 Days Forecast for {symbol}')
    plt.legend()
    plt.grid(True)
    print("well does it?")
    plt.show()

# Interactive prompt
while True:
    user_input = input(f"=====Enter a company symbol from the available options (e.g., {available_symbols[0]})======: ").strip()
    try:
        symbol = int(user_input)
        if symbol in available_symbols:
            predict_stock(symbol)
            break
        else:
            print("Invalid input. Please enter a valid company symbol.")
    except ValueError:
        print("Invalid input. Please enter a numeric company symbol.")

    # Ask if the user wants to try another symbol
    continue_input = input("Do you want to try another symbol? (yes/no): ").strip().lower()
    if continue_input != 'yes':
        print("Exiting program.")
        break
