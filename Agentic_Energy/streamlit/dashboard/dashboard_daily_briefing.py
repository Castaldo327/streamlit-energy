import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os

# Initialize session state for data
if 'data_initialized' not in st.session_state:
    st.session_state.data_initialized = False

def load_data_from_csv():
    """Load historical and forecast data directly from CSV files"""
    data_dir = "data"
    historical_file = os.path.join(data_dir, "charlottesville_historical_data.csv")
    forecast_file = os.path.join(data_dir, "charlottesville_weather_forecast.csv")
    
    # Check if files exist
    if not os.path.exists(historical_file):
        st.error(f"Historical data file not found at {historical_file}. Please generate the data first.")
        return None, None
    
    if not os.path.exists(forecast_file):
        st.warning(f"Forecast data file not found at {forecast_file}. Only historical data will be available.")
        forecast_data = pd.DataFrame()
    else:
        # Load forecast data
        forecast_data = pd.read_csv(forecast_file)
        forecast_data['timestamp'] = pd.to_datetime(forecast_data['timestamp'])
    
    # Load historical data
    historical_data = pd.read_csv(historical_file)
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    return historical_data, forecast_data

def initialize_data():
    """Initialize or refresh the data in the session state"""
    if not st.session_state.data_initialized:
        # First load
        st.session_state.cville_data, st.session_state.forecast_data = load_data_from_csv()
        st.session_state.data_initialized = True
        st.session_state.last_update = datetime.now()
    else:
        # Check if it's time to refresh data (e.g., every hour)
        time_since_update = datetime.now() - st.session_state.last_update
        if time_since_update > timedelta(hours=1):
            st.session_state.cville_data, st.session_state.forecast_data = load_data_from_csv()
            st.session_state.last_update = datetime.now()

def dashboard_daily_briefing():
    st.subheader("Today's Energy Overview for Charlottesville")

    # Initialize data by reading directly from CSV files
    initialize_data()
    
    # Check if data was loaded successfully
    if st.session_state.cville_data is None:
        st.error("Failed to load data. Please make sure the data files exist.")
        return
    
    # Get today's data
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_data = st.session_state.cville_data[st.session_state.cville_data['timestamp'].dt.date == today.date()]
    
    if len(today_data) == 0:
        st.warning("No data available for today. Showing most recent day instead.")
        most_recent_date = st.session_state.cville_data['timestamp'].dt.date.max()
        today_data = st.session_state.cville_data[st.session_state.cville_data['timestamp'].dt.date == most_recent_date]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_load = today_data['load_mw'].iloc[-1] if not today_data.empty else 0
        avg_load = today_data['load_mw'].mean() if not today_data.empty else 0
        delta = ((current_load / avg_load) - 1) * 100 if avg_load > 0 else 0
        st.metric("Current Load (MW)", f"{current_load:.1f}", f"{delta:.1f}%")
    
    with col2:
        current_price = today_data['lmp_price'].iloc[-1] if not today_data.empty else 0
        avg_price = today_data['lmp_price'].mean() if not today_data.empty else 0
        delta = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
        st.metric("Current LMP ($/MWh)", f"${current_price:.2f}", f"{delta:.1f}%")
    
    with col3:
        renewable_pct = ((today_data['solar_generation'].iloc[-1] + today_data['wind_generation'].iloc[-1]) / 
                         today_data['load_mw'].iloc[-1] * 100) if not today_data.empty and today_data['load_mw'].iloc[-1] > 0 else 0
        st.metric("Renewable %", f"{renewable_pct:.1f}%")
    
    with col4:
        congestion = today_data['congestion_index'].iloc[-1] if not today_data.empty else 0
        st.metric("Congestion Index", f"{congestion:.1f}/100")
    
    # Daily load and price chart
    st.subheader("Today's Load and Price Profile")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=today_data['timestamp'],
        y=today_data['load_mw'],
        name='Load (MW)',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=today_data['timestamp'],
        y=today_data['lmp_price'],
        name='LMP Price ($/MWh)',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Load and Price for Charlottesville Area',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Load (MW)', side='left'),
        yaxis2=dict(title='Price ($/MWh)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generation mix
    st.subheader("Today's Generation Mix")

    today_data['timestamp'] = pd.to_datetime(today_data['timestamp'])

    # Check for missing or misaligned data
    if today_data[['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation']].isnull().values.any():
        st.error("There are missing values in the generation data.")
    else:
        # Ensure all generation columns are correctly aligned
        generation_data = today_data[['timestamp', 'solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation']]
        
        fig = px.area(
            generation_data, 
            x='timestamp', 
            y=['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation'],
            labels={'value': 'Generation (MW)', 'variable': 'Source'},
            title='Generation Mix',
            color_discrete_map={
                'solar_generation': 'gold',
                'wind_generation': 'skyblue',
                'natural_gas_generation': 'gray',
                'other_generation': 'darkgreen'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Daily briefing notes
    st.subheader("Daily Intelligence Briefing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Market Insights")
        
        # Generate some synthetic insights based on the data
        max_price_hour = today_data.loc[today_data['lmp_price'].idxmax(), 'timestamp'].hour if not today_data.empty else 0
        max_load_hour = today_data.loc[today_data['load_mw'].idxmax(), 'timestamp'].hour if not today_data.empty else 0
        
        avg_price = today_data['lmp_price'].mean() if not today_data.empty else 0
        yesterday = today - timedelta(days=1)
        yesterday_data = st.session_state.cville_data[st.session_state.cville_data['timestamp'].dt.date == yesterday.date()]
        yesterday_avg_price = yesterday_data['lmp_price'].mean() if not yesterday_data.empty else avg_price
        
        price_change = ((avg_price / yesterday_avg_price) - 1) * 100 if yesterday_avg_price > 0 else 0
        
        st.info(f"• Peak prices occurred at hour ending {max_price_hour+1}:00")
        st.info(f"• Peak load occurred at hour ending {max_load_hour+1}:00")
        st.info(f"• Average LMP is {price_change:.1f}% {'higher' if price_change > 0 else 'lower'} than yesterday")
        
        if not today_data.empty and today_data['congestion_index'].max() > 70:
            st.warning("• High congestion detected during peak hours")
        
    with col2:
        st.markdown("#### Recommended Actions")
        
        # Generate recommendations based on data
        if avg_price > 50:
            st.warning("• Consider load reduction during peak price hours")
        else:
            st.success("• Prices are favorable for normal operations")
            
        if renewable_pct > 20:
            st.success("• High renewable penetration today - good opportunity for clean energy credits")
        else:
            st.info("• Low renewable generation today - monitor fossil fuel prices")
            
        # Add a weather-related recommendation
        current_temp = today_data['temperature'].iloc[-1] if not today_data.empty else 0
        if current_temp > 85:
            st.warning("• High temperatures may drive afternoon peak prices")
        elif current_temp < 40:
            st.warning("• Cold temperatures may drive morning and evening peak prices")

# For testing the dashboard function directly
if __name__ == "__main__":
    dashboard_daily_briefing()