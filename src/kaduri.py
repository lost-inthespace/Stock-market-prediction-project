import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#====== docu ======
# creating the LSTM model and training it, then saving it.

# Load the dataset
data_path = 'Equites_Historical_Adjusted_Prices_Report.csv'
df = pd.read_csv(data_path)

# Convert 'Date' to datetime and sort values
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date'])

# Select relevant columns for analysis
data = df[['Symbol', 'Date', 'Close', 'Volume Traded']]

# Normalize the 'Close' prices
scalers = {}

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60
X_all, y_all = [], []

# Prepare data for all companies
#This loop iterates over each unique Symbol in the dataset:

    # Extracts Date and Close columns for the specific symbol and sets Date as the index.
    # Initializes a MinMaxScaler and scales the Close prices to a range between 0 and 1.
    # Stores the scaler in the scalers dictionary.
    # Calls create_sequences() to generate input-output pairs.
    # If valid sequences are generated (X.shape[0] > 0), they are appended to X_all and y_all.
for symbol in data['Symbol'].unique():
    symbol_data = data[data['Symbol'] == symbol][['Date', 'Close']].set_index('Date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(symbol_data['Close'].values.reshape(-1, 1))
    scalers[symbol] = scaler
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Check the shape of the created sequences
    if X.shape[0] > 0:  # Only append if there are valid sequences
        X_all.append(X)
        y_all.append(y)

# Stack arrays only if they have valid shapes
X_all = np.vstack(X_all) if X_all else np.array([])
y_all = np.concatenate(y_all) if y_all else np.array([])

# Check if X_all and y_all are not empty before proceeding
if X_all.size == 0 or y_all.size == 0:
    raise ValueError("No valid data to train the model. Please check the data preparation step.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=False)

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
history = model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test, y_test))

# Save the trained model
model.save('stock_price_model.h5')

print("=====    All is done, evaluations are next.  =====\n")

# Plotting the loss graph
print("===  plotting the loss graph ===")
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


#printing evaluations
# Predict on the test set
print("predicting.....")
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("prediction done, evaluations as follows:")
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R-squared: {r2:.4f}')

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Prices', marker='o')
plt.plot(predictions, label='Predicted Prices', marker='x')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#latest training evaluations:
# Mean Squared Error: 0.0006
# Mean Absolute Error: 0.0177
# R-squared: 0.9889