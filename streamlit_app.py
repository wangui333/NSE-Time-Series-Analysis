import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import plotly.express as px

# Load feature data
DATA_PATH = 'nse_features.csv'
df = pd.read_csv(DATA_PATH)
stocks = df['Code'].unique()

# Streamlit UI
st.title("📈 NSE Stock Price Predictor")

selected_stock = st.selectbox("Choose a stock ticker:", sorted(stocks))

# Filter and sort data
stock_data = df[df['Code'] == selected_stock].copy()
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
stock_data = stock_data.sort_values(by='Date')

# Define cutoff date
N_DAYS = 30
latest_date = stock_data['Date'].max()
cutoff_date = latest_date - pd.Timedelta(days=N_DAYS)

# Filter for last N days and aggregate by date
trend_data = (
    stock_data[stock_data['Date'] >= cutoff_date]
    .groupby('Date', as_index=False)['Day Price']
    .mean()  # handles multiple entries per date
)

# Layout: Tabs for Trend and Prediction
tab1, tab2 = st.tabs(["📊 Recent Price Trend", "🤖 Predict Next Price"])

# Tab 1: Price Trend
with tab1:
    st.subheader(f"{selected_stock} - Interactive Price Trend")

    # Slider to choose number of days
    max_days = (stock_data['Date'].max() - stock_data['Date'].min()).days
    N_DAYS = st.slider("Select how many past days to show", min_value=7, max_value=max(30, max_days), value=30)

    # Filter trend data
    latest_date = stock_data['Date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=N_DAYS)
    trend_data = stock_data[stock_data['Date'] >= cutoff_date][['Date', 'Day Price']].dropna()

    if not trend_data.empty:
        fig = px.line(
            trend_data,
            x='Date',
            y='Day Price',
            title=f'{selected_stock} Price Trend (Last {N_DAYS} Days)',
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Day Price (KES)",
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=14, label="2w", step="day", stepmode="backward"),
                        dict(count=30, label="1m", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough valid data to display a trend.")

# Tab 2: Prediction
with tab2:
    st.subheader("Predict Tomorrow's Price")
    if st.button("Predict"):
        try:
            # Load model and scaler
            model_path = f"models/{selected_stock}.keras"
            scaler_path = f"models/{selected_stock}_scaler.gz"
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)

            # Prepare features
            features = ['12m Low', '12m High', 'Day Low', 'Day High', 'Previous',
                        'Change', 'Change%', 'Volume', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI']
            latest_data = stock_data[-10:]

            if len(latest_data) < 10:
                st.warning("Not enough recent data for prediction.")
            else:
                X_input = scaler.transform(latest_data[features])
                X_input = np.expand_dims(X_input, axis = 0)
                y_log_pred = model.predict(X_input)[0][0]
                y_pred = np.expm1(y_log_pred)

                st.success(f"Predicted Day Price: **{y_pred:.2f} KES**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")