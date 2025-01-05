import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model

# Load the data
data = pd.read_csv('Equites_Historical_Adjusted_Prices_Report.csv')

# Convert 'Date' column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])
data.set_index('Date', inplace=True)

# Sort by date
data = data.sort_index()

# Encode company names
encoder = OneHotEncoder()
company_encoded = encoder.fit_transform(data[['Company Name']]).toarray()

# Define function to prepare data
def prepare_data(df, company_encoded, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    X_prices, X_companies, y = [], [], []
    for i in range(look_back, len(scaled_values)):
        X_prices.append(scaled_values[i - look_back:i, 0])
        X_companies.append(company_encoded[i])
        y.append(scaled_values[i, 0])
    
    return np.array(X_prices), np.array(X_companies), np.array(y), scaler

# Prepare data for all companies
X_prices, X_companies, y, scaler = prepare_data(data, company_encoded)

# Reshape price data for LSTM input
X_prices = X_prices.reshape((X_prices.shape[0], X_prices.shape[1], 1))

# Create LSTM model
input_prices = Input(shape=(X_prices.shape[1], 1))
input_company = Input(shape=(X_companies.shape[1],))

x = LSTM(units=50, return_sequences=True)(input_prices)
x = LSTM(units=50, return_sequences=False)(x)

# Concatenate company info with LSTM output
x = Concatenate()([x, input_company])
output = Dense(units=1)(x)

model = Model(inputs=[input_prices, input_company], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_prices, X_companies], y, epochs=5, batch_size=32, verbose=1)

# Save the model
model.save('combined_stock_prediction_model.h5')
print("Model saved successfully.")

# Example prediction logic for a specific company
def predict_for_company(company_name, look_back=60, n_days=7):
    company_index = encoder.transform([[company_name]]).toarray()
    
    # Use the last sequence from the dataset
    last_sequence = X_prices[-1].reshape(1, -1, 1)
    future_predictions = []

    for _ in range(n_days):
        next_day_prediction = model.predict([last_sequence, company_index], verbose=0)[0, 0]
        unscaled_prediction = scaler.inverse_transform([[next_day_prediction]])[0, 0]
        future_predictions.append(unscaled_prediction)
        
        # Update the sequence with the new prediction
        last_sequence = np.append(last_sequence[:, 1:, :], next_day_prediction).reshape(1, -1, 1)
    
    return future_predictions

# Predict for a specific company
predictions = predict_for_company('Riyadh Bank')
print(predictions)
