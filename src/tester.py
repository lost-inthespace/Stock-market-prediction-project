#this code will funtion to pull the result data from other training sessions without running them again
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Path to the saved model and new data
model_path = os.path.join(os.path.dirname(__file__), '../resources/stock_lstm_model.h5')
new_data_path = os.path.join(os.path.dirname(__file__), '../resources/stock_data.csv')

# Load the trained model
model = load_model(model_path)

# Load new data
new_data = pd.read_csv(new_data_path)
new_data = new_data.sort_values('التاريخ')

# Feature columns
features = ['إفتتاح', 'إقفال', 'الأدنى', 'التغيير', '% التغيير']

# Scale the data (use the same scaler used during training)
scaler = MinMaxScaler()
new_data[features] = scaler.fit_transform(new_data[features])

# Prepare sequences
def create_sequences(data, sequence_length=60):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][features].values)
    return np.array(X)

sequence_length = 60
X_new = create_sequences(new_data)

# Make predictions
predictions = model.predict(X_new)

# Print or save the results
print("Predictions:")
print(predictions)

# Optionally save the predictions to a CSV
output_path = os.path.join(os.path.dirname(__file__), '../resources/predictions.csv')
pd.DataFrame(predictions, columns=['Predicted Closing Price']).to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
