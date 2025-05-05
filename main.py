from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load the LSTM model
model = tf.keras.models.load_model("lstm_model.h5")

# Create FastAPI app
app = FastAPI(title = "LSTM Stock Price Predictor")

# Define input schema
class StockFeatures(BaseModel):
    Low_12m: float
    High_12m: float
    Day_Low: float
    Day_High: float
    Previous: float
    Change: float
    Change_percent: float
    Volume: float
    SMA_10: float
    SMA_50: float
    EMA_10: float
    EMA_50: float
    RSI: float

@app.post("/predict")
def predict_price(features: StockFeatures):
    # Convert input to model-ready format
    input_data = np.array([
        [
            features.Low_12m,
            features.High_12m,
            features.Day_Low,
            features.Day_High,
            features.Previous,
            features.Change,
            features.Change_percent,
            features.Volume,
            features.SMA_10,
            features.SMA_50,
            features.EMA_10,
            features.EMA_50,
            features.RSI
        ]
    ])

    # Reshape for LSTM: (samples, timesteps, features)
    input_data = input_data.reshape((1, 1, 13))

    # Make prediction
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0][0])}