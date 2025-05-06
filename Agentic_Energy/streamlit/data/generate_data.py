import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fetch_weather_data(start_date, end_date, location="Charlottesville, VA"):
    # Placeholder function to fetch historical weather data from an API
    # Replace with actual API calls to sources like NOAA or Weather Underground
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    weather_data = []
    for timestamp in date_range:
        weather_data.append({
            'timestamp': timestamp,
            'temperature': np.random.normal(60, 10),  # Replace with actual temperature data
            'humidity': np.random.normal(50, 10),     # Replace with actual humidity data
            'wind_speed': np.random.normal(5, 2),     # Replace with actual wind speed data
            'solar_radiation': np.random.normal(200, 50)  # Replace with actual solar radiation data
        })
    return pd.DataFrame(weather_data)

def generate_charlottesville_data(days=1825):
    """Generate synthetic power market data relevant to Charlottesville, VA"""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Fetch historical weather data
    weather_data = fetch_weather_data(start_date, end_date)
    
    # Base load pattern for Charlottesville area (smaller city)
    base_load_mw = 150  # Base load in MW for the Charlottesville area
    
    data = []
    for timestamp in date_range:
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0-6 where 0 is Monday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Get weather data for the current timestamp
        weather = weather_data.loc[weather_data['timestamp'] == timestamp].iloc[0]
        temperature = weather['temperature']
        humidity = weather['humidity']
        wind_speed = weather['wind_speed']
        solar_radiation = weather['solar_radiation']
        
        # Load calculation
        seasonal_load = base_load_mw * (1 + 0.3 * np.sin((timestamp.dayofyear / 365) * 2 * np.pi))
        hourly_pattern = 1 + 0.2 * np.sin((hour - 8) / 24 * 2 * np.pi) + 0.3 * np.sin((hour - 18) / 24 * 2 * np.pi)
        weekend_factor = 0.8 if is_weekend else 1.0
        temp_factor = 1 + 0.2 * ((temperature - 65) / 30)**2
        
        # Calculate final load
        load = seasonal_load * hourly_pattern * weekend_factor * temp_factor * np.random.normal(1, 0.05)
        
        # PJM LMP pricing for Dominion zone
        base_price = 35  # Base price in $/MWh
        price_season = base_price * (1 + 0.2 * np.sin((timestamp.dayofyear / 365) * 2 * np.pi))
        price_hour = price_season * (1 + 0.5 * np.sin((hour - 16) / 24 * 2 * np.pi))
        price_load = price_hour * (1 + 0.3 * (load / (base_load_mw * 1.3) - 0.7))
        lmp_price = max(10, price_load * np.random.normal(1, 0.15))
        
        # Generation mix data
        solar_capacity = 30  # MW of solar in Charlottesville area
        wind_capacity = 10   # MW of wind in Charlottesville area
        
        # Solar generation
        solar_factor = max(0, np.sin((hour - 6) / 12 * np.pi)) if 6 <= hour <= 18 else 0
        cloud_factor = np.random.uniform(0.3, 1.0)
        solar_generation = solar_capacity * solar_factor * cloud_factor
        
        # Wind generation
        wind_factor = np.random.normal(0.3, 0.15)
        wind_generation = wind_capacity * max(0, wind_factor)
        
        # Natural gas and other generation
        natural_gas = max(0, load - solar_generation - wind_generation) * 0.7
        other_generation = max(0, load - solar_generation - wind_generation - natural_gas)
        
        # Congestion indicator
        congestion = np.random.gamma(2, 10) if load > base_load_mw * 1.5 else np.random.gamma(1, 5)
        congestion = min(100, congestion)
        
        data.append({
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation,
            'load_mw': load,
            'lmp_price': lmp_price,
            'solar_generation': solar_generation,
            'wind_generation': wind_generation,
            'natural_gas_generation': natural_gas,
            'other_generation': other_generation,
            'is_weekend': is_weekend,
            'congestion_index': congestion,
            'day_of_week': day_of_week
        })
    
    return pd.DataFrame(data)

def generate_weather_forecast(forecast_days=10):
    """Generate a 10-day weather forecast for Charlottesville, VA"""
    # Create date range for forecast
    start_date = datetime.now()
    end_date = start_date + timedelta(days=forecast_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    forecast_data = []
    for timestamp in date_range:
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0-6 where 0 is Monday
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Generate synthetic weather data
        season_factor = np.sin((timestamp.dayofyear / 365) * 2 * np.pi)
        base_temp = 60 + 20 * season_factor  # Base temperature varies seasonally
        daily_temp_variation = 15 * np.sin((hour - 14) / 24 * 2 * np.pi)  # Peak at 2 PM
        temperature = base_temp + daily_temp_variation + np.random.normal(0, 3)  # Add some noise
        humidity = np.random.normal(50, 10)  # Synthetic humidity data
        wind_speed = np.random.normal(5, 2)  # Synthetic wind speed data
        solar_radiation = np.random.normal(200, 50)  # Synthetic solar radiation data
        
        forecast_data.append({
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'solar_radiation': solar_radiation,
            'is_weekend': is_weekend,
            'day_of_week': day_of_week
        })
    
    return pd.DataFrame(forecast_data)

def main():
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Generate historical data
    print("Generating historical data...")
    historical_data = generate_charlottesville_data(days=365)
    
    # Save historical data to CSV
    historical_file = os.path.join(data_dir, "charlottesville_historical_data.csv")
    historical_data.to_csv(historical_file, index=False)
    print(f"Historical data saved to {historical_file} ({len(historical_data)} records)")
    
    # Generate weather forecast data
    print("Generating weather forecast data...")
    weather_forecast = generate_weather_forecast(forecast_days=10)
    
    # Save weather forecast data to CSV
    forecast_file = os.path.join(data_dir, "charlottesville_weather_forecast.csv")
    weather_forecast.to_csv(forecast_file, index=False)
    print(f"Weather forecast data saved to {forecast_file} ({len(weather_forecast)} records)")
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()