import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from market.price_analysis import render_price_analysis
from market.load_analysis import render_load_analysis
from market.generation_mix import render_generation_mix
from market.correlation_explorer import render_correlation_explorer
from market.heat_maps import render_heat_maps
from data.data_loader import load_data

def market_analysis_main():
    st.subheader("Market Analysis Tools")

    # Load data
    forecast_data, historical_data = load_data()
    
    # Combine forecast and historical data
    st.session_state.cville_data = pd.concat([historical_data, forecast_data], ignore_index=True)

    # Create tabs for different analysis tools
    analysis_tabs = st.tabs([
        "Price Analysis", 
        "Load Analysis", 
        "Generation Mix", 
        "Correlation Explorer",
        "Heat Maps"
    ])
    
    # Date range selection (shared across tabs)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    # Filter data based on date range
    mask = (st.session_state.cville_data['timestamp'].dt.date >= start_date) & \
           (st.session_state.cville_data['timestamp'].dt.date <= end_date)
    filtered_data = st.session_state.cville_data[mask]
    
    # Render each tab with the filtered data
    with analysis_tabs[0]:
        render_price_analysis(filtered_data)
    
    with analysis_tabs[1]:
        render_load_analysis(filtered_data)
    
    with analysis_tabs[2]:
        render_generation_mix(filtered_data, start_date, end_date)
    
    with analysis_tabs[3]:
        render_correlation_explorer(filtered_data)
    
    with analysis_tabs[4]:
        render_heat_maps(filtered_data)