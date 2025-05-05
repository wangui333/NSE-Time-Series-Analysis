from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = FastAPI()

# Constants
TIME_STEPS = 10
FEATURES = ['12m Low', '12m High', 'Day Low', 'Day High', 'Previous',
            'Change', 'Change%', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']

class PredictionRequest(BaseModel):
    stock_code: str

def load_model_and_scaler(stock_code):
    try:
        model = load_model(f"models/{stock_code}.keras")

        scaler = MinMaxScaler()
        scaler.scale_ = np.load(f"models/{stock_code}_scaler.npy")
        scaler.min_ = np.load(f"models/{stock_code}_min.npy")

        return model, scaler
    except Exception:
        raise HTTPException(status_code = 404, detail = f"No model found for stock: {stock_code}")

def create_input_sequence(data, scaler):
    data_scaled = scaler.transform(data[FEATURES])
    prices_log = np.log1p(data['Day Price'].values.reshape(-1, 1))
    combined = np.hstack((data_scaled, prices_log))
    X = np.array([combined[-TIME_STEPS:, :-1]])
    return X

@app.post("/predict")
def predict_price(request: PredictionRequest):
    stock_code = request.stock_code.upper()

    # Load latest data
    try:
        df = pd.read_csv("nse_features.csv")
        stock_df = df[df['Code'] == stock_code].copy()
    except:
        raise HTTPException(status_code = 500, detail = "Could not read latest data.")

    if len(stock_df) < TIME_STEPS:
        raise HTTPException(status_code = 400, detail = "Not enough historical data for prediction.")

    # Load model + scaler
    model, scaler = load_model_and_scaler(stock_code)

    # Create input and predict
    try:
        X_input = create_input_sequence(stock_df, scaler)
        log_pred = model.predict(X_input)[0][0]
        pred_price = np.expm1(log_pred)

        return {
            "stock_code": stock_code,
            "predicted_price": round(float(pred_price), 2),
            "log_prediction": round(float(log_pred), 4),
            "last_known_price": float(stock_df.iloc[-1]['Day Price'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Prediction failed: {e}")