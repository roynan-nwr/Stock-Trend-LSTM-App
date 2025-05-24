# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

st.title("📈 Stock Price Analyzer & LSTM Predictor")

# --- Sidebar inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

if st.sidebar.button("Run Analysis"):
    # --- Fetch Data ---
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the selected ticker and date range. Please try again.")
    else:
        data.reset_index(inplace=True)
        close_data = data[['Date', 'Close']]

        # --- Preprocessing ---
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(close_data[['Close']])

        # --- Sequence creation ---
        def create_sequences(data, seq_length=50):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_close)

        if len(X) == 0:
            st.warning("Not enough data points to create sequences for LSTM. Try increasing the date range.")
        else:
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # --- LSTM Model ---
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            predicted = model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted)
            real_prices = scaler.inverse_transform(y.reshape(-1, 1))

            # Align predictions
            prediction_dates = close_data['Date'][-len(predicted_prices):].values

            # --- Interactive Plot ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prediction_dates, y=real_prices.flatten(), mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted'))
            fig.update_layout(title=f"{ticker} Price Prediction with LSTM",
                              xaxis_title="Date",
                              yaxis_title="Price")

            st.plotly_chart(fig, use_container_width=True)
            st.success("Analysis complete! Use the sidebar to tweak inputs.")
