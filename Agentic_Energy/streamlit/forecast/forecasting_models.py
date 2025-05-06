import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# Example data preprocessing function
def preprocess_data_for_forecasting(data):
    df = data.copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Create time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Drop rows with NaN values created by lag/rolling features
    df.dropna(inplace=True)
    
    return df

def train_load_forecast_model(data, algorithm="random_forest"):
    df = preprocess_data_for_forecasting(data)
    
    if df.empty or len(df) < 50:
        st.warning("Not enough data to train the load forecast model.")
        return None, {}, []
    
    load_features = [
        'temperature', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
    ]
    
    X = df[load_features]
    y = df['load_mw']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if algorithm == "linear_regression":
        model = LinearRegression()
    elif algorithm == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif algorithm == "neural_network":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
    
    model.fit(X_train, y_train)
    
    if algorithm == "neural_network":
        X_test_scaled = model.scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = model.scaler_y.inverse_transform(y_pred_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    
    return model, metrics, load_features

def train_price_forecast_model(data, algorithm="random_forest"):
    df = preprocess_data_for_forecasting(data)
    
    if df.empty or len(df) < 50:
        st.warning("Not enough data to train the price forecast model.")
        return None, {}, []
    
    price_features = [
        'temperature', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend'
    ]
    
    X = df[price_features]
    y = df['lmp_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    if algorithm == "linear_regression":
        model = LinearRegression()
    elif algorithm == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif algorithm == "neural_network":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
    
    model.fit(X_train, y_train)
    
    if algorithm == "neural_network":
        X_test_scaled = model.scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = model.scaler_y.inverse_transform(y_pred_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    
    return model, metrics, price_features

def evaluate_model_performance(model, X, y, model_type="load"):
    if hasattr(model, 'scaler_X'):
        X_scaled = model.scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = model.scaler_y.inverse_transform(y_pred_scaled).flatten()
    else:
        y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    r2 = r2_score(y, y_pred)
    
    performance_df = pd.DataFrame({
        'actual': y,
        'predicted': y_pred,
        'error': y - y_pred,
        'abs_error': np.abs(y - y_pred),
        'pct_error': np.abs((y - y_pred) / y) * 100
    })
    
    metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
    
    return metrics, performance_df

def get_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return dict(zip(features, importance))
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        return dict(zip(features, importance))
    else:
        return {feature: 1 / len(features) for feature in features}

def generate_forecasts(forecast_data, load_model, price_model, load_features, price_features):
    forecast_with_predictions = forecast_data.copy()
    
    if 'timestamp' in forecast_with_predictions.columns:
        forecast_with_predictions.set_index('timestamp', inplace=True)
    
    forecast_with_predictions['hour'] = forecast_with_predictions.index.hour
    forecast_with_predictions['hour_sin'] = np.sin(2 * np.pi * forecast_with_predictions['hour'] / 24)
    forecast_with_predictions['hour_cos'] = np.cos(2 * np.pi * forecast_with_predictions['hour'] / 24)
    
    # Generate load forecast
    try:
        X_load = forecast_with_predictions[load_features]
        
        if hasattr(load_model, 'scaler_X'):
            X_load_scaled = load_model.scaler_X.transform(X_load)
            load_pred_scaled = load_model.predict(X_load_scaled)
            forecast_with_predictions['load_forecast'] = load_model.scaler_y.inverse_transform(load_pred_scaled).flatten()
        else:
            forecast_with_predictions['load_forecast'] = load_model.predict(X_load)
    except Exception as e:
        print(f"Error generating load forecast: {e}")
        forecast_with_predictions['load_forecast'] = 50 + 10 * np.sin(2 * np.pi * forecast_with_predictions['hour'] / 24)
    
    forecast_with_predictions['load_mw'] = forecast_with_predictions['load_forecast']
    
    # Generate price forecast
    try:
        X_price = forecast_with_predictions[price_features]
        
        if hasattr(price_model, 'scaler_X'):
            X_price_scaled = price_model.scaler_X.transform(X_price)
            price_pred_scaled = price_model.predict(X_price_scaled)
            forecast_with_predictions['price_forecast'] = price_model.scaler_y.inverse_transform(price_pred_scaled).flatten()
        else:
            forecast_with_predictions['price_forecast'] = price_model.predict(X_price)
    except Exception as e:
        print(f"Error generating price forecast: {e}")
        forecast_with_predictions['price_forecast'] = 20 + 0.2 * forecast_with_predictions['load_forecast']
    
    forecast_with_predictions['price_confidence'] = 95 - np.arange(len(forecast_with_predictions)) * 0.5
    
    forecast_with_predictions.reset_index(inplace=True)
    
    return forecast_with_predictions
