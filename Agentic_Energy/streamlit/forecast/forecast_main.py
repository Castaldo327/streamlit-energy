import streamlit as st
import pandas as pd
import numpy as np
from data.data_utils import load_data
from forecast.forecasting_models import train_load_forecast_model, train_price_forecast_model, generate_forecasts
from forecast.ui_components import display_load_forecast, display_price_forecast
from forecast.ui_components_model_performance import display_model_performance
from forecast.ui_components_custom_model import display_custom_model_builder
from forecast.ui_components_scenario import display_scenario_analysis


# Main function for forecasting and modeling
def forecasting_and_modeling():
    st.subheader("Energy Forecasting Models for Charlottesville")
    
    forecast_tabs = st.tabs([
        "Load Forecast", 
        "Price Forecast", 
        "Model Performance", 
        "Custom Model Builder",
        "Scenario Analysis"
    ])
    
    # Load data
    historical_data = load_data()
    
    # Use historical data only
    st.session_state.cville_data = historical_data
    
    # Check if models exist in session state, otherwise train them
    if 'load_model' not in st.session_state:
        with st.spinner("Training load forecast model..."):
            load_model, load_metrics, load_features = train_load_forecast_model(st.session_state.cville_data)
            st.session_state.load_model = load_model
            st.session_state.load_metrics = load_metrics
            st.session_state.load_features = load_features
    
    if 'price_model' not in st.session_state:
        with st.spinner("Training price forecast model..."):
            price_model, price_metrics, price_features = train_price_forecast_model(st.session_state.cville_data)
            st.session_state.price_model = price_model
            st.session_state.price_metrics = price_metrics
            st.session_state.price_features = price_features
    
    # Check if we need to update forecasts (either they don't exist, they're missing columns, or update was requested)
    update_needed = (
        'forecast_data' not in st.session_state or 
        'load_forecast' not in st.session_state.forecast_data.columns or
        st.session_state.get('update_forecasts', False)
    )
    
    if update_needed:
        with st.spinner("Generating forecasts..."):
            # Create a forecast period (e.g., next 72 hours)
            last_timestamp = st.session_state.cville_data['timestamp'].max()
            forecast_start = pd.to_datetime(last_timestamp) + pd.Timedelta(hours=1)
            forecast_end = forecast_start + pd.Timedelta(hours=72)
            
            # Create a DataFrame with hourly timestamps
            forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='h')
            forecast_df = pd.DataFrame({'timestamp': forecast_timestamps})
            
            # Add basic features
            forecast_df['hour'] = forecast_df['timestamp'].dt.hour
            forecast_df['day_of_week'] = forecast_df['timestamp'].dt.dayofweek
            forecast_df['is_weekend'] = forecast_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add weather features (in a real app, you'd get these from a weather API)
            # For now, we'll use synthetic data
            forecast_df['temperature'] = 70 + 10 * np.sin(2 * np.pi * (forecast_df['hour'] - 12) / 24)
            forecast_df['humidity'] = 50 + 10 * np.cos(2 * np.pi * forecast_df['hour'] / 24)
            forecast_df['wind_speed'] = 5 + 3 * np.random.rand(len(forecast_df))
            forecast_df['solar_radiation'] = np.maximum(0, 500 * np.sin(np.pi * (forecast_df['hour'] - 6) / 12))
            forecast_df.loc[forecast_df['hour'] < 6, 'solar_radiation'] = 0
            forecast_df.loc[forecast_df['hour'] > 18, 'solar_radiation'] = 0
            
            # Determine which models to use
            if st.session_state.get('use_custom_load_model', False):
                load_model = st.session_state.custom_load_model
                load_features = st.session_state.custom_load_features
            else:
                load_model = st.session_state.load_model
                load_features = st.session_state.load_features

            if st.session_state.get('use_custom_price_model', False):
                price_model = st.session_state.custom_price_model
                price_features = st.session_state.custom_price_features
            else:
                price_model = st.session_state.price_model
                price_features = st.session_state.price_features
            
            # Generate forecasts using the selected models
            st.session_state.forecast_data = generate_forecasts(
                forecast_df,
                load_model,
                price_model,
                load_features,
                price_features
            )
            
            # Reset the update flag
            st.session_state.update_forecasts = False
            
            st.success("Forecasts updated successfully!")
    
    # Tab 1: Load Forecast
    with forecast_tabs[0]:
        display_load_forecast(st.session_state.cville_data, st.session_state.forecast_data)
    
    # Tab 2: Price Forecast
    with forecast_tabs[1]:
        display_price_forecast(st.session_state.cville_data, st.session_state.forecast_data)
    
    # Tab 3: Model Performance
    with forecast_tabs[2]:
        display_model_performance(st.session_state.cville_data, 
                                  st.session_state.load_model, 
                                  st.session_state.price_model,
                                  st.session_state.load_features,
                                  st.session_state.price_features)
    
    # Tab 4: Custom Model Builder
    with forecast_tabs[3]:
        display_custom_model_builder(st.session_state.cville_data)
    
    # Tab 5: Scenario Analysis
    with forecast_tabs[4]:
        display_scenario_analysis(st.session_state.forecast_data)

# Run the forecasting and modeling function
if __name__ == "__main__":
    forecasting_and_modeling()