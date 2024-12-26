# code written by mohammad Kashkari 

print("just testing")
# importing stuff here
import os
import pandas as pand
import numpy as numbs
import matplotlib.pyplot as plotter

#checking versions
print("pandas version: "+pand.__version__)
print("numpy ver: "+ numbs.__version__)

# ========begin the main code here========


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Path to the data file
data_path = os.path.join(os.path.dirname(__file__), '../resources/stock_data.csv')

# Load the data
data = pand.read_csv(data_path)

# Ensure data is sorted by date
data = data.sort_values('التاريخ')

# Feature columns and target column
features = ['إفتتاح', 'إقفال', 'الأدنى', 'التغيير', '% التغيير']
target = 'إقفال'

# Scale the data
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare sequences
def create_sequences(data, target_column, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][features].values)
        y.append(data.iloc[i + sequence_length][target_column])
    return numbs.array(X), numbs.array(y)

sequence_length = 60  # Use the past 60 days of data to predict
X, y = create_sequences(data, target)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save the trained model
model.save(os.path.join(os.path.dirname(__file__), '../resources/stock_lstm_model.h5'))

# Display training results
plotter.figure(figsize=(12, 6))
plotter.plot(history.history['loss'], label='Training Loss')
plotter.plot(history.history['val_loss'], label='Validation Loss')
plotter.title('Model Loss During Training')
plotter.xlabel('Epochs')
plotter.ylabel('Loss')
plotter.legend()
plotter.grid()
plotter.show()

print("Model training complete and saved.")
