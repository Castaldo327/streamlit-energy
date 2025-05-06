import pandas as pd

def load_data():
    """Load data from CSV files and preprocess it."""
    # Load data from CSV files
    forecast_data = pd.read_csv('data/charlottesville_weather_forecast.csv')
    historical_data = pd.read_csv('data/charlottesville_historical_data.csv')
    

    forecast_data['timestamp'] = pd.to_datetime(forecast_data['timestamp'])
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    return forecast_data, historical_data