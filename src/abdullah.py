import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Equites_Historical_Adjusted_Prices_Report.csv')

# Convert 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y', errors='coerce')

# Drop rows with invalid dates
data = data.dropna(subset=['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Ensure the data is sorted by Date (important for future predictions)
data = data.sort_index()

# Process each company
def prepare_data(df, column='Close', look_back=60):
    values = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(look_back, len(scaled_values)):
        X.append(scaled_values[i - look_back:i, 0])  # Use the previous `look_back` days
        y.append(scaled_values[i, 0])  # Predict the next day's closing price
        
    return np.array(X), np.array(y), scaler

# Create LSTM model
def create_lstm_model(input_shape):
    #type of LSTM model
    model = Sequential()

    #first layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    
    #second layers
    model.add(LSTM(units=50, return_sequences=False))
    #third layer
    model.add(Dense(units=1))  # Single output (closing price)
    
    #compile and finish the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and predict closing price for the next 7 days
def predict_closing_price(data, n_days=7, look_back=60):
    for company in data['Company Name'].unique():
        print(f"\n================Processing company: {company}================")
        
        # Prepare data for the company
        company_data = data[data['Company Name'] == company]
        X, y, scaler = prepare_data(company_data, look_back=look_back)
        
        # Reshape data for LSTM (3D shape)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Create and train the model
        model = create_lstm_model((X.shape[1], 1))
        #this runs and fits the model, it runs over every company
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)  # Increase epochs

        # Predict next 7 days
        last_sequence = X[-1].reshape(1, -1, 1)  # Use the last available sequence
        future_predictions = []
        future_dates = []

        last_date = company_data.index[-1]  # Get the last date from the dataset

        # Set the future dates (consecutive days after the last date in the dataset)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D').strftime('%d/%m/%Y')

        for i in range(n_days):
            next_day_prediction = model.predict(last_sequence, verbose=0)[0, 0]
            unscaled_prediction = scaler.inverse_transform([[next_day_prediction]])[0, 0]
            future_predictions.append(unscaled_prediction)
            
            # Update the sequence with the new prediction
            last_sequence = np.append(last_sequence[:, 1:, :], next_day_prediction).reshape(1, -1, 1)
        
        # Display predictions with corresponding dates
        print(f"\nPredicted Closing Prices for {company}:")
        for date, price in zip(future_dates, future_predictions):
            print(f"{date}: {price:.2f} SAR")

# Run the prediction
predict_closing_price(data)
