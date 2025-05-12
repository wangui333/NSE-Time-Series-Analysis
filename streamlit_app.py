import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import zipfile

# Load feature data

@st.cache_data
def load_data():
    with zipfile.ZipFile("nse_features_full.zip") as z:
        with z.open("nse_features_full.csv") as f:
            df = pd.read_csv(f)
    return df

df = load_data()


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
    .mean()
)

# Layout: Tabs for Trend and Prediction
tab1, tab2, tab3 = st.tabs(["Recent Price Trend", "Predict Next Price", "Investment Risk"])

# Tab 1: Price Trend
with tab1:
    st.subheader(f"{selected_stock} - Interactive Price Trend")

    max_days = (stock_data['Date'].max() - stock_data['Date'].min()).days
    N_DAYS = st.slider("Select how many past days to show", min_value=7, max_value=max(30, max_days), value=30)

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
    
    with st.expander("📈 Trend Classification (Last 30 Days)", expanded=True):
        recent = stock_data.copy().sort_values(by='Date').tail(40).dropna()

    if recent.shape[0] < 30:
        st.warning("Not enough recent data to analyze trend.")
    else:
        # Compute EMAs
        recent['EMA_10'] = recent['Day Price'].ewm(span=10, adjust=False).mean()
        recent['EMA_30'] = recent['Day Price'].ewm(span=30, adjust=False).mean()

        # Classify trend
        latest_ema10 = recent['EMA_10'].iloc[-1]
        latest_ema30 = recent['EMA_30'].iloc[-1]

        if latest_ema10 > latest_ema30 * 1.01:
            trend = "🟢 Bullish (Uptrend)"
            color = "green"
        elif latest_ema10 < latest_ema30 * 0.99:
            trend = "🔴 Bearish (Downtrend)"
            color = "red"
        else:
            trend = "🟡 Neutral (Sideways)"
            color = "orange"

        st.markdown(f"**Trend:** <span style='color:{color}; font-size: 20px;'>{trend}</span>", unsafe_allow_html=True)

        # Plot trend
        fig_trend = px.line(
            recent, x='Date', y=['Day Price', 'EMA_10', 'EMA_30'],
            labels={"value": "Price (KES)", "variable": "Line"},
            title=f"{selected_stock} Trend with EMA10 & EMA30"
        )
        fig_trend.update_traces(mode='lines+markers')
        st.plotly_chart(fig_trend, use_container_width=True)

# Tab 2: Prediction
with tab2:
    st.subheader("Predict Tomorrow's Price")
    if st.button("Predict"):
        try:
            model_path = f"saved_xgb_models/xgb_model_{selected_stock}.pkl"
            model = joblib.load(model_path)

            # Prepare latest lag features
            feature_cols = [
                'Day Price_lag1', 'Day Price_lag2', 'Day Price_lag3',
                'RSI_lag1', 'RSI_lag2', 'RSI_lag3',
                'EMA_10_lag1', 'EMA_10_lag2', 'EMA_10_lag3'
            ]
            latest_row = stock_data.dropna(subset=feature_cols).sort_values('Date').iloc[-1]

            input_features = latest_row[feature_cols].values.reshape(1, -1)
            prediction = model.predict(input_features)[0]

            st.success(f"Predicted Day Price: **{prediction:.2f} KES**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.subheader("📉 RSI Indicator")

    if 'RSI' in stock_data.columns:
        rsi_data = stock_data[['Date', 'RSI']].dropna()
        rsi_data['Date'] = pd.to_datetime(rsi_data['Date'])

        # Filter RSI to last N_DAYS only
        rsi_data = rsi_data[rsi_data['Date'] >= cutoff_date].copy()

        if rsi_data.empty:
            st.warning("Not enough RSI data for the recent period.")
        else:
            # Detect buy/sell signals based on RSI crossing thresholds
            rsi_data = rsi_data.sort_values('Date').reset_index(drop=True)
            rsi_data['Signal'] = None

            for i in range(1, len(rsi_data)):
                prev = rsi_data.loc[i - 1, 'RSI']
                curr = rsi_data.loc[i, 'RSI']
                if prev < 30 and curr >= 30:
                    rsi_data.loc[i, 'Signal'] = 'Buy'
                elif prev > 70 and curr <= 70:
                    rsi_data.loc[i, 'Signal'] = 'Sell'

            # Plot RSI with threshold lines and signals
            fig_rsi = px.line(
                rsi_data,
                x='Date',
                y='RSI',
                title=f"{selected_stock} - RSI Indicator (Last {N_DAYS} Days)",
                markers=True
            )

            fig_rsi.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="Oversold", annotation_position="top left")
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="green", annotation_text="Overbought", annotation_position="top left")

            # Add signal markers
            signals = rsi_data.dropna(subset=['Signal'])
            fig_rsi.add_scatter(
                x=signals['Date'],
                y=signals['RSI'],
                mode='markers+text',
                marker=dict(color='orange', size=10, symbol='triangle-up'),
                text=signals['Signal'],
                textposition='top center',
                name='Signal'
            )

            fig_rsi.update_layout(
                xaxis_title="Date",
                yaxis_title="RSI",
                yaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig_rsi, use_container_width=True)

    else:
        st.warning("RSI data not found for this stock.")

# Tab 3: Risk Analysis
with tab3:
    st.subheader("📊 Investment Risk Level")

    if stock_data.shape[0] < 30:
        st.warning("Not enough data to evaluate risk.")
    else:
        # Sort and compute returns
        stock_data = stock_data.sort_values(by='Date')
        stock_data['Return'] = stock_data['Day Price'].pct_change()

        # Compute CV for current stock
        mean_price = stock_data['Day Price'].mean()
        std_price = stock_data['Day Price'].std()
        risk_score = std_price / mean_price if mean_price != 0 else np.nan

        # Compute CVs for all stocks
        cv_dict = {}
        for stock in stocks:
            subset = df[df['Code'] == stock]
            if subset.shape[0] < 30:
                continue
            mean = subset['Day Price'].mean()
            std = subset['Day Price'].std()
            if mean != 0:
                cv_dict[stock] = std / mean

        cv_series = pd.Series(cv_dict)
        low_thresh = cv_series.quantile(0.33)
        high_thresh = cv_series.quantile(0.66)

        # Risk label
        if risk_score <= low_thresh:
            label = "🟢 Safe Investment (Low Volatility)"
            recommendation = "This stock shows stable pricing and may be suitable for conservative investors."
        elif risk_score <= high_thresh:
            label = "🟡 Moderate Risk"
            recommendation = "This stock has moderate volatility. Good for balanced portfolios."
        else:
            label = "🔴 Risky Investment (High Volatility)"
            recommendation = "This stock is highly volatile. Caution advised unless you have a high risk appetite."

        # Show score and label
        st.metric(label="Risk Score (CV)", value=f"{risk_score:.4f}")
        st.caption(f"Thresholds — Low ≤ {low_thresh:.4f}, Moderate ≤ {high_thresh:.4f}, High > {high_thresh:.4f}")
        st.info(label)
        st.write(recommendation)

        # Gauge chart for CV
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            delta = {'reference': cv_series.median(), 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [0, max(cv_series.max(), 0.2)]},
                'steps': [
                    {'range': [0, low_thresh], 'color': "lightgreen"},
                    {'range': [low_thresh, high_thresh], 'color': "khaki"},
                    {'range': [high_thresh, cv_series.max()], 'color': "salmon"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk_score}
            },
            title={'text': "Risk Gauge (CV)"}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Histogram of CVs
        fig_hist = px.histogram(
            cv_series,
            nbins=20,
            title="Distribution of Risk Scores (CV) Across All Stocks",
            labels={'value': 'CV Score', 'count': 'Number of Stocks'}
        )
        fig_hist.add_vline(x = risk_score, line_color = 'blue', line_dash = 'dash', annotation_text = "Selected Stock", annotation_position = "top")
        st.plotly_chart(fig_hist, use_container_width=True)
