import joblib
import json
from flask import Flask, request, jsonify
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    model = joblib.load('housing_model.joblib')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Housing Price Prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on the input features"""
    try:
        # Get input features from request
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")
        
        # Convert input data to model features format
        features = np.array([[
            float(data.get('MedInc', {}).get('0', 0)),
            float(data.get('AveRooms', {}).get('0', 0)),
            float(data.get('AveBedrms', {}).get('0', 0)),
            float(data.get('Population', {}).get('0', 0)),
            float(data.get('AveOccup', {}).get('0', 0)),
            float(data.get('Latitude', {}).get('0', 0))
        ]])
        logging.info(f"the extracted features are: {features}")
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction as JSON
        return jsonify({
            'prediction': prediction[0],
            'features': data
        })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)