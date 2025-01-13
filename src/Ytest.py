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
    X = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
    return np.array(X)

# Function to fetch stock data for the past week using yfinance
def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    data = stock_data.history(period='5d')
    return data[['Close']]

# Function to predict the next week's prices
def predict_next_week(symbol, sequence_length=50):
    # Match symbol with company name
    company_row = df[df['Symbol'] == symbol]
    # if company_row.empty:
    #     print(f"Symbol {symbol} not found in CSV.")
    #     return
    
    #@@@@@   this is not entirly correct, it doesnt show the correct info     @@@@@
    company_name = company_row['Company Name']
    print("#" * 5 ,f"Predicting for: {company_name} ({symbol})", "#" * 5)
    
    # Create the ticker symbol for Saudi companies
    #@@@@@  this is how saudi tickers are, the number symbol+.SR  @@@@@
    ticker = f"{symbol}.SR"
    
    # Fetch the last week's data
    #@@@    check correctness   @@@
    stock_data = fetch_stock_data(ticker)
    
    # Check if data is fetched successfully
    if stock_data.empty:
        print(f"No data fetched for {symbol}.")
        return
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    # Create sequences for prediction
    #@@@@@  this seems to create issues, need to fix @@@@@
    last_sequence = create_sequences(data_scaled, sequence_length)
    
    # Reshape to fit the LSTM input requirements
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    
    # Predict the next 7 days
    future_predictions = []
    for _ in range(7):
        next_day_prediction = model.predict(last_sequence)[0, 0]
        unscaled_prediction = scaler.inverse_transform([[next_day_prediction]])[0, 0]
        future_predictions.append(unscaled_prediction)
        
        # Update the sequence with the new prediction
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_day_prediction]], axis=1)
    
    # Return the predicted values
    return stock_data.index[-1], future_predictions

# Plotting function
def plot_predictions(start_date, predictions, symbol):
    future_dates = [start_date + pd.Timedelta(days=i+1) for i in range(7)]
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, predictions, marker='o', linestyle='-', label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Predicted Closing Price')
    plt.title(f'Predicted Closing Prices for {symbol}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example symbol to predict
    #testing for riyad bank
    example_symbol = '6012'  # Replace with an actual symbol from your CSV
    start_date, predictions = predict_next_week(example_symbol)
    
    if predictions:
        plot_predictions(start_date, predictions, example_symbol)
