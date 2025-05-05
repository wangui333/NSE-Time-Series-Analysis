import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Load feature data
DATA_PATH = 'nse_features.csv'
df = pd.read_csv(DATA_PATH)
stocks = df['Code'].unique()

# Streamlit UI
st.title("NSE Stock Price Predictor")
selected_stock = st.selectbox("Choose a stock ticker:", sorted(stocks))

if st.button("Predict"):
    try:
        # Load the saved model and scaler
        model_path = f"models/{selected_stock}.keras"
        scaler_path = f"models/{selected_stock}_scaler.gz"
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Prepare latest 10 rows of features for the selected stock
        stock_data = df[df['Code'] == selected_stock].sort_values(by = 'Date')[-10:]
        features = ['12m Low', '12m High', 'Day Low', 'Day High', 'Previous',
                    'Change', 'Change%', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']
        X_input = scaler.transform(stock_data[features])
        X_input = np.expand_dims(X_input, axis = 0)

        # Predict
        y_log_pred = model.predict(X_input)[0][0]
        y_pred = np.expm1(y_log_pred)

        st.success(f"Predicted Day Price: **{y_pred:.2f} KES**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")