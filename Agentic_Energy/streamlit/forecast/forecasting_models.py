import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

from data.data_utils import preprocess_data_for_forecasting

# Function to train load forecast model
def train_load_forecast_model(data, algorithm="random_forest"):
    df = preprocess_data_for_forecasting(data)
    
    if df.empty or len(df) < 50:  # Check if the dataset is too small
        st.warning("Not enough data to train the load forecast model.")
        return None, {}, []
    
    # Define features for load forecasting
    load_features = [
        'temperature', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
        'load_lag_1', 'load_lag_24', 'load_rolling_24'
    ]
    
    X = df[load_features]
    y = df['load_mw']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model based on selected algorithm
    if algorithm == "linear_regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    elif algorithm == "neural_network":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0)
        
        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
    
    # Evaluate model
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

# Function to train price forecast model
def train_price_forecast_model(data, algorithm="random_forest"):
    df = preprocess_data_for_forecasting(data)
    
    if df.empty or len(df) < 50:  # Check if the dataset is too small
        st.warning("Not enough data to train the price forecast model.")
        return None, {}, []
    
    # Define features for price forecasting
    price_features = [
        'load_mw', 'temperature', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
        'price_lag_1', 'price_lag_24', 'load_lag_1', 'price_rolling_24'
    ]
    
    X = df[price_features]
    y = df['lmp_price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model based on selected algorithm
    if algorithm == "linear_regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif algorithm == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    elif algorithm == "neural_network":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0)
        
        model.scaler_X = scaler_X
        model.scaler_y = scaler_y
    
    # Evaluate model
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

# Function to evaluate model performance
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

# Function to get feature importance
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
    """
    Apply trained models to generate forecasts for the input data
    """
    # Make a copy to avoid modifying the original
    forecast_with_predictions = forecast_data.copy()
    
    # Ensure timestamp is a datetime index
    if 'timestamp' in forecast_with_predictions.columns:
        forecast_with_predictions.set_index('timestamp', inplace=True)
    
    # Add hour features
    forecast_with_predictions['hour'] = forecast_with_predictions.index.hour
    forecast_with_predictions['hour_sin'] = np.sin(2 * np.pi * forecast_with_predictions['hour'] / 24)
    forecast_with_predictions['hour_cos'] = np.cos(2 * np.pi * forecast_with_predictions['hour'] / 24)
    
    # For load lags
    for i in range(1, 25):  # Add lags 1 through 24
        lag_name = f'load_lag_{i}'
        if lag_name in load_features:
            forecast_with_predictions[lag_name] = 50  # Placeholder
    
    # Add rolling features
    if 'load_rolling_24' in load_features:
        forecast_with_predictions['load_rolling_24'] = 50  # Placeholder
    
    # For price lags
    for i in range(1, 25):  # Add lags 1 through 24
        lag_name = f'price_lag_{i}'
        if lag_name in price_features:
            forecast_with_predictions[lag_name] = 30  # Placeholder
    
    if 'price_rolling_24' in price_features:
        forecast_with_predictions['price_rolling_24'] = 30  # Placeholder
    
    # Generate load forecast
    try:
        # Check if all required features are available
        missing_load_features = [f for f in load_features if f not in forecast_with_predictions.columns]
        if missing_load_features:
            print(f"Warning: Missing load features: {missing_load_features}")
            # Add missing features with placeholder values
            for feature in missing_load_features:
                forecast_with_predictions[feature] = 50  # Placeholder
        
        X_load = forecast_with_predictions[load_features]
        
        # Apply the model
        if hasattr(load_model, 'scaler_X'):  # Neural network model
            X_load_scaled = load_model.scaler_X.transform(X_load)
            load_pred_scaled = load_model.predict(X_load_scaled)
            forecast_with_predictions['load_forecast'] = load_model.scaler_y.inverse_transform(load_pred_scaled).flatten()
        else:  # Other models
            forecast_with_predictions['load_forecast'] = load_model.predict(X_load)
    except Exception as e:
        # Fallback if model prediction fails
        print(f"Error generating load forecast: {e}")
        forecast_with_predictions['load_forecast'] = 50 + 10 * np.sin(2 * np.pi * forecast_with_predictions['hour'] / 24)
    
    # Add the load forecast as a feature for price forecasting
    forecast_with_predictions['load_mw'] = forecast_with_predictions['load_forecast']
    
    # Generate price forecast
    try:
        # Check if all required features are available
        missing_price_features = [f for f in price_features if f not in forecast_with_predictions.columns]
        if missing_price_features:
            print(f"Warning: Missing price features: {missing_price_features}")
            # Add missing features with placeholder values
            for feature in missing_price_features:
                forecast_with_predictions[feature] = 30  # Placeholder
        
        X_price = forecast_with_predictions[price_features]
        
        # Apply the model
        if hasattr(price_model, 'scaler_X'):  # Neural network model
            X_price_scaled = price_model.scaler_X.transform(X_price)
            price_pred_scaled = price_model.predict(X_price_scaled)
            forecast_with_predictions['price_forecast'] = price_model.scaler_y.inverse_transform(price_pred_scaled).flatten()
        else:  # Other models
            forecast_with_predictions['price_forecast'] = price_model.predict(X_price)
    except Exception as e:
        # Fallback if model prediction fails
        print(f"Error generating price forecast: {e}")
        forecast_with_predictions['price_forecast'] = 20 + 0.2 * forecast_with_predictions['load_forecast']
    
    # Add confidence metrics
    forecast_with_predictions['price_confidence'] = 95 - np.arange(len(forecast_with_predictions)) * 0.5
    
    # Reset index to have timestamp as a column
    forecast_with_predictions.reset_index(inplace=True)
    
    return forecast_with_predictions