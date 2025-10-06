#!/usr/bin/env python
# coding: utf-8

# In[68]:


# dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import logging
import matplotlib.pyplot as plt


# ------------------------
# Suppress noisy logs
# ------------------------
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("statsmodels").setLevel(logging.ERROR)

st.set_page_config(page_title="KPI Forecasting Dashboard", layout="wide")

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Dissertation_df.csv", parse_dates=["study_date"])
    return df

df = load_data()

# ------------------------
# Prepare KPIs per modality
# ------------------------
def prepare_kpis(df):
    kpis = df.groupby(["study_date", "modality"]).agg(
        avg_tat=("turnaround_time_days", "mean"),
        avg_report_tat=("report_turnaround_time_days", "mean"),
        avg_schedule_delays=("schedule_delay_days", "mean"),
        daily_scan_volume=("daily_scan_volume", "mean"),
        daily_report_volume=("daily_report_volume", "mean")
    ).reset_index()
    return kpis

daily_kpis_modality = prepare_kpis(df)
# ------------------------
# Forecasting Functions
# ------------------------
def forecast_arima(series, forecast_days):
    try:
        # Convert to Series if it's not already and ensure frequency is set
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Set the frequency
        series = series.asfreq('D')
        
        # Fit ARIMA model with more conservative parameters (p,d,q)
        # Using (1,1,1) as a more stable starting point
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Create proper date range for forecast
        last_date = series.index[-1]
        forecast_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_days, 
                                     freq='D')
        
        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_days)
        return pd.Series(forecast, index=forecast_idx)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")
        return pd.Series()

def forecast_prophet(series, forecast_days):
    try:
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        
        # Fit Prophet model
        model = Prophet()
        model.fit(df)
        
        # Create future dataframe
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_days, 
                                     freq='D')
        future = pd.DataFrame({'ds': future_dates})
        
        # Make forecast
        forecast = model.predict(future)
        
        # Return as Series with proper date index
        return pd.Series(forecast['yhat'].values, index=future_dates)
    except Exception as e:
        st.warning(f"Prophet failed: {e}")
        return pd.Series()

def forecast_lstm(series, forecast_days):
    try:
        # Normalize data
        values = series.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        # Parameters
        seq_length = 10
        X, y = create_sequences(scaled, seq_length)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build and train LSTM model using the recommended approach
        model = Sequential()
        # Use Input layer explicitly as recommended
        model.add(Input(shape=(seq_length, 1)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, verbose=0)
        
        # Forecast
        last_sequence = scaled[-seq_length:].reshape(1, seq_length, 1)
        forecast_scaled = []
        
        for _ in range(forecast_days):
            next_pred = model.predict(last_sequence, verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[:, 1:, :], 
                                     next_pred.reshape(1, 1, 1), 
                                     axis=1)
        
        # Inverse transform
        forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
        
        # Create proper date range for forecast
        last_date = series.index[-1]
        forecast_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=forecast_days, 
                                     freq='D')
        
        return pd.Series(forecast_values.flatten(), index=forecast_idx)
    except Exception as e:
        st.warning(f"LSTM failed: {e}")
        return pd.Series()

# ------------------------
# Streamlit UI
# ------------------------
st.title("📊 Radiology KPI Forecasting Dashboard")

# Assuming daily_kpis_modality is loaded elsewhere in your code
modality = st.selectbox("Select Modality", daily_kpis_modality["modality"].unique())
forecast_days = st.slider("Forecast horizon (days)", 7, 90, 30)

# Get data for selected modality and ensure proper datetime index
df_mod = daily_kpis_modality[daily_kpis_modality["modality"]==modality].copy()
df_mod["study_date"] = pd.to_datetime(df_mod["study_date"])
df_mod = df_mod.set_index("study_date").sort_index()

# Set the frequency to daily
df_mod = df_mod.asfreq('D')

# Fill missing values if needed after setting frequency - using ffill() instead of fillna(method='ffill')
df_mod = df_mod.ffill()

kpi_columns = [col for col in df_mod.columns if col != "modality"]

tabs = st.tabs(kpi_columns)

for tab, kpi in zip(tabs, kpi_columns):
    with tab:
        st.subheader(f"{kpi} Forecast ({modality})")

        series = df_mod[kpi].dropna()
        
        # Ensure series has frequency set
        series = series.asfreq('D')

        # Forecasts
        arima_forecast = forecast_arima(series, forecast_days)
        prophet_forecast = forecast_prophet(series, forecast_days)
        lstm_forecast = forecast_lstm(series, forecast_days)

        # Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines+markers', name='History'))

        # Add forecasts to plot if they're not empty
        if not arima_forecast.empty:
            fig.add_trace(go.Scatter(x=arima_forecast.index, y=arima_forecast.values, mode='lines', name='ARIMA Forecast'))
        if not prophet_forecast.empty:
            fig.add_trace(go.Scatter(x=prophet_forecast.index, y=prophet_forecast.values, mode='lines', name='Prophet Forecast'))
        if not lstm_forecast.empty:
            fig.add_trace(go.Scatter(x=lstm_forecast.index, y=lstm_forecast.values, mode='lines', name='LSTM Forecast'))

        fig.update_layout(title=f"{kpi} Forecast for {modality}",
                          xaxis_title="Date",
                          yaxis_title=kpi,
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key=f"{modality}_{kpi}_chart")


# In[ ]:




