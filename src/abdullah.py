import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
data = pd.read_csv('Equites_Historical_Adjusted_Prices_Report.csv')

# Convert 'Date' column to datetime
#vital step to chose our time range, prob gonna need to see how to adjust it later
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Drop rows with invalid dates
data = data.dropna(subset=['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Debugging: Check date range in the dataset
print("Date range in the dataset:")
print(f"Start: {data.index.min()}, End: {data.index.max()}")

# Process each company
companies = data['Company Name'].unique()
all_filtered_data = pd.DataFrame()

for company in companies:
    print(f"======== Processing company: {company} ==========")
    
    # Filter data for the current company
    company_data = data[data['Company Name'] == company]
    
    # Sort by date (if not already sorted)
    company_data = company_data.sort_index()
    
    # Calculate date range for the last 3 years
    end_date = company_data.index.max()
    start_date = end_date - pd.DateOffset(years=3)
    
    # Ensure the calculated range exists in the data
    if start_date < company_data.index.min():
        print(f"Start date {start_date} is before the available data for {company}. Adjusting...")
        start_date = company_data.index.min()
    
    # Filter data for the last 3 years
    filtered_data = company_data.loc[start_date:end_date]
    print(f"Filtered data range for {company}: {filtered_data.index.min()} to {filtered_data.index.max()}")
    
    # Append to a combined dataframe
    all_filtered_data = pd.concat([all_filtered_data, filtered_data])

# Prepare the data for LSTM
def prepare_data(df, column='Close', look_back=60):
    values = df[column].values.reshape(-1, 1)
    scaled_values = (values - values.min()) / (values.max() - values.min())  # Min-max scaling
    
    X, y = [], []
    for i in range(look_back, len(scaled_values)):
        X.append(scaled_values[i-look_back:i, 0])
        y.append(scaled_values[i, 0])
    return np.array(X), np.array(y)

# Example for a single company
example_company_data = all_filtered_data[all_filtered_data['Company Name'] == companies[0]]
X_train, y_train = prepare_data(example_company_data)

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save('stock_prediction_model.h5')
print("Model saved successfully.")
