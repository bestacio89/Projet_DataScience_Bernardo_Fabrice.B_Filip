# diabetes_prediction_api.py

"""
Diabetes Prediction API

This script creates a FastAPI-based RESTful API to serve predictions from the
trained Perceptron (Neural Network), Random Forest, and Linear Regression models.
Clients can send data to the API and receive predictions.

Author: Bernardo Estacio Abreu, Fabrice Bellin, Filip Dabrowsky
Date: 15/11/2024
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained models
rf_model = joblib.load("model/diabete/RandomForestModel.pkl")
lr_model = joblib.load("model/diabete/LinearRegressionModel.pkl")
perceptron_model = tf.keras.models.load_model("model/diabete/DiabetePerceptron.keras")

# Define the input schema for the API
class PredictionInput(BaseModel):
    features: list  # List of feature values for prediction

# Define a health check endpoint
@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict: API status.
    """
    return {"status": "API is running"}

# Endpoint for Neural Network (Perceptron) predictions
@app.post("/predict/perceptron")
async def predict_perceptron(input_data: PredictionInput):
    """
    Predict using the Perceptron (Neural Network) model.

    Parameters:
        input_data (PredictionInput): Input features for prediction.

    Returns:
        dict: Predicted value.
    """
    features = np.array(input_data.features).reshape(1, -1)
    prediction = perceptron_model.predict(features)[0][0]
    return {"model": "Perceptron", "prediction": prediction}

# Endpoint for Random Forest predictions
@app.post("/predict/randomforest")
async def predict_randomforest(input_data: PredictionInput):
    """
    Predict using the Random Forest model.

    Parameters:
        input_data (PredictionInput): Input features for prediction.

    Returns:
        dict: Predicted value.
    """
    features = np.array(input_data.features).reshape(1, -1)
    prediction = rf_model.predict(features)[0]
    return {"model": "Random Forest", "prediction": prediction}

# Endpoint for Linear Regression predictions
@app.post("/predict/linearreg")
async def predict_linearreg(input_data: PredictionInput):
    """
    Predict using the Linear Regression model.

    Parameters:
        input_data (PredictionInput): Input features for prediction.

    Returns:
        dict: Predicted value.
    """
    features = np.array(input_data.features).reshape(1, -1)
    prediction = lr_model.predict(features)[0]
    return {"model": "Linear Regression", "prediction": prediction}
