import pandas as pd
import os

def load_data():
    """Load data from CSV files and preprocess it."""
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load historical data
    historical_data_path = os.path.join(current_dir, 'charlottesville_historical_data.csv')
    historical_data = pd.read_csv(historical_data_path)
    
    # Load forecast data
    forecast_data_path = os.path.join(current_dir, 'charlottesville_weather_forecast.csv')
    forecast_data = pd.read_csv(forecast_data_path)
    
    # Convert timestamp columns to datetime
    forecast_data['timestamp'] = pd.to_datetime(forecast_data['timestamp'])
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    return forecast_data, historical_data
