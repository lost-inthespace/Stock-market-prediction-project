# simple of using flask as an backend for html to see the result you should 
# run this code and open the corresponded html is the same folder
# the result of the predection will print in the html screen

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'stock_lstm_model.h5')
model = tf.keras.models.load_model(model_path)

# Initialize MinMaxScaler (same as used during training)
scaler = MinMaxScaler()
features = ['إفتتاح', 'إقفال', 'الأدنى', 'التغيير', '% التغيير']

# Function to preprocess and create sequences
def preprocess_data(data, sequence_length=60):
    data[features] = scaler.fit_transform(data[features])  # Normalize features
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i + sequence_length][features].values)
    return np.array(sequences)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Example: Fetch stock data for "بنك الرياض"
        # Replace with actual fetching logic (API or database query)
        stock_data = pd.read_csv("stock_data.csv")

        # Ensure enough data for sequence creation
        if len(stock_data) < 60:
            return jsonify({"error": "Not enough data to make a prediction. At least 60 days required."}), 400

        # Preprocess data
        sequences = preprocess_data(stock_data)

        # Predict the next price
        prediction = model.predict(sequences[-1].reshape(1, -1, len(features)))

        return jsonify({"predicted_price": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
