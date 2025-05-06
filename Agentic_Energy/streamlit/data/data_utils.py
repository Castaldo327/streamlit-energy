import pandas as pd
import numpy as np

# Load data function
def load_data():
    historical_data = pd.read_csv('data/charlottesville_historical_data.csv')
    
    # Convert timestamp columns to datetime
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    return historical_data

# Preprocess data function
def preprocess_data_for_forecasting(data):
    df = data.copy()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create cyclical features for hour of day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Create lag features for load and price
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'load_lag_{lag}'] = df['load_mw'].shift(lag)
        df[f'price_lag_{lag}'] = df['lmp_price'].shift(lag)
    
    # Create rolling mean features
    for window in [3, 6, 12, 24]:
        df[f'load_rolling_{window}'] = df['load_mw'].rolling(window=window).mean()
        df[f'price_rolling_{window}'] = df['lmp_price'].rolling(window=window).mean()
    
    # Drop rows with NaN values (from lag/rolling features)
    df = df.dropna()
    
    return df