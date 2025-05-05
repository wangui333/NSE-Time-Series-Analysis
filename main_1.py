from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load preprocessed feature dataset
nse_df = pd.read_csv("nse_features.csv")

# Define model input features
FEATURES = ['12m Low', '12m High', 'Day Low', 'Day High', 'Previous',
            'Change', 'Change%', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']

# Fit the same scaler used during training
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(nse_df[FEATURES])

# Load trained LSTM model
model = tf.keras.models.load_model("lstm_model.h5")

class TickerInput(BaseModel):
    ticker: str

@app.post("/predict")
def predict_price(payload: TickerInput):
    ticker = payload.ticker.upper()
    df = nse_df[nse_df["Code"] == ticker].copy()
    
    if df.empty or len(df) < 10:
        raise HTTPException(status_code = 400, detail = "Not enough data for this ticker.")

    df = df.sort_values("Date")
    df = df.dropna(subset = FEATURES)
    
    if len(df) < 10:
        raise HTTPException(status_code = 400, detail = "Not enough valid rows after dropping NaNs.")

    # Prepare last 10 rows
    recent = df.iloc[-10:][FEATURES]
    recent_scaled = scaler.transform(recent)
    input_seq = recent_scaled.reshape(1, 10, len(FEATURES))

    # Predict log price → convert back to real price
    log_pred = model.predict(input_seq)
    price_pred = np.expm1(log_pred[0][0])

    return {
        "ticker": ticker,
        "predicted_price": round(float(price_pred), 2)
    }