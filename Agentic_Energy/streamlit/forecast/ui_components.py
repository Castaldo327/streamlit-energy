import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Function to display load forecast tab
def display_load_forecast(historical_data, forecast_data):
    st.subheader("Load Forecasting")
    
    # Create copies to avoid modifying the originals
    historical_data = historical_data.copy()
    forecast_data = forecast_data.copy()
    
    # Ensure timestamp is properly handled
    if 'timestamp' in historical_data.columns:
        historical_data.set_index('timestamp', inplace=True)
    if 'timestamp' in forecast_data.columns:
        forecast_data.set_index('timestamp', inplace=True)
    
    # Ensure datetime index is present
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = pd.to_datetime(historical_data.index)
    if not isinstance(forecast_data.index, pd.DatetimeIndex):
        forecast_data.index = pd.to_datetime(forecast_data.index)
    
    # Forecast horizon selection
    forecast_hours = st.slider(
        "Forecast Horizon (hours)", 
        1, 72, 24, 
        key="load_forecast_horizon"  # Add this unique key
    )
    
    # Get historical data for context
    historical_hours = 24  # Show last 24 hours of historical data
    recent_data = historical_data.sort_index().tail(historical_hours)
    
    # Get forecast data
    forecast_data = forecast_data.head(forecast_hours)
    
    # Combine for visualization
    combined_timestamps = list(recent_data.index) + list(forecast_data.index)
    combined_load = list(recent_data['load_mw']) + [None] * len(forecast_data)
    combined_forecast = [None] * len(recent_data) + list(forecast_data['load_forecast'])
    
    # Create confidence intervals for the forecast
    upper_bound = [None] * len(recent_data) + [
        forecast_data['load_forecast'].iloc[i] * (1 + (0.05 * (i+1) / 24))
        for i in range(len(forecast_data))
    ]
    
    lower_bound = [None] * len(recent_data) + [
        forecast_data['load_forecast'].iloc[i] * (1 - (0.05 * (i+1) / 24))
        for i in range(len(forecast_data))
    ]
    
    # Create the plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=combined_load,
        mode='lines',
        name='Historical Load',
        line=dict(color='blue', width=3)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=combined_forecast,
        mode='lines',
        name='Load Forecast',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        showlegend=False
    ))
    
    # Add vertical line to separate historical from forecast
    forecast_start_timestamp = recent_data.index[-1]
    
    if hasattr(forecast_start_timestamp, 'timestamp'):
        forecast_start_timestamp = forecast_start_timestamp.timestamp() * 1000
    else:
        forecast_start_timestamp = datetime.timestamp(forecast_start_timestamp) * 1000
        
    fig.add_vline(
        x=forecast_start_timestamp,
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Load Forecast for Charlottesville',
        xaxis_title='Time',
        yaxis_title='Load (MW)',
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast details table
    st.markdown("### Forecast Details")
    
    # Create a dataframe with forecast details
    forecast_details = forecast_data[['load_forecast']].copy()
    if 'temperature' in forecast_data.columns:
        forecast_details['temperature'] = forecast_data['temperature']
    
    # Add the index as a column
    forecast_details['Time'] = forecast_details.index
    
    # Rename columns
    forecast_details = forecast_details.rename(columns={
        'load_forecast': 'Load Forecast (MW)',
        'temperature': 'Temperature Forecast (°F)'
    })
    
    # Format the columns
    forecast_details['Time'] = forecast_details['Time'].dt.strftime('%Y-%m-%d %H:%M')
    forecast_details['Load Forecast (MW)'] = forecast_details['Load Forecast (MW)'].round(1)
    if 'Temperature Forecast (°F)' in forecast_details.columns:
        forecast_details['Temperature Forecast (°F)'] = forecast_details['Temperature Forecast (°F)'].round(1)
    
    st.dataframe(forecast_details, use_container_width=True)
    
    # Forecast drivers
    st.markdown("### Forecast Drivers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Temperature Impact")
        if 'temperature' in forecast_data.columns:
            fig = px.scatter(
                forecast_data,
                x='temperature',
                y='load_forecast',
                labels={
                    'temperature': 'Forecast Temperature (°F)',
                    'load_forecast': 'Forecast Load (MW)'
                },
                title='Temperature vs. Load Forecast'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Temperature data is not available for visualization")
    
    with col2:
        st.markdown("#### Hourly Pattern")
        
        hourly_forecast = forecast_data.groupby(forecast_data.index.hour)['load_forecast'].mean().reset_index()
        hourly_forecast.columns = ['Hour', 'Average Load Forecast (MW)']
        
        fig = px.bar(
            hourly_forecast,
            x='Hour',
            y='Average Load Forecast (MW)',
            title='Hourly Load Forecast Pattern',
            labels={
                'Hour': 'Hour of Day',
                'Average Load Forecast (MW)': 'Average Load (MW)'
            }
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_price_forecast(historical_data, forecast_data):
    st.subheader("Price Forecasting")
    
    # Create copies to avoid modifying the originals
    historical_data = historical_data.copy()
    forecast_data = forecast_data.copy()
    
    # We'll use the same forecast horizon as load forecast
    forecast_hours = st.slider("Forecast Horizon (hours)", 1, 72, 24, key="price_forecast_horizon")
    
    # Get historical price data
    historical_hours = 24  # Show last 24 hours of historical data
    
    # Handle the case where timestamp is either a column or an index
    if 'timestamp' in historical_data.columns:
        recent_price_data = historical_data.sort_values('timestamp').tail(historical_hours)
    else:
        # If timestamp is the index
        historical_data = historical_data.copy()
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            historical_data.index = pd.to_datetime(historical_data.index)
        recent_price_data = historical_data.sort_index().tail(historical_hours)
    
    # Get forecast data and ensure it's properly formatted
    forecast_data = forecast_data.head(forecast_hours)
    
    # Handle the case where timestamp is either a column or an index
    if 'timestamp' in forecast_data.columns:
        # Keep timestamp as a column
        forecast_timestamps = list(forecast_data['timestamp'])
    else:
        # If timestamp is the index
        forecast_timestamps = list(forecast_data.index)
    
    if 'timestamp' in recent_price_data.columns:
        historical_timestamps = list(recent_price_data['timestamp'])
    else:
        historical_timestamps = list(recent_price_data.index)
    
    # Combine for visualization
    combined_timestamps = historical_timestamps + forecast_timestamps
    
    # Get load values
    if 'lmp_price' in recent_price_data.columns:
        combined_price = list(recent_price_data['lmp_price']) + [None] * len(forecast_data)
    else:
        # If recent_price_data is indexed by timestamp and doesn't have a timestamp column
        st.error("Historical data doesn't have an 'lmp_price' column. Please check your data.")
        return
    
    # Get forecast values
    combined_price_forecast = [None] * len(recent_price_data) + list(forecast_data['price_forecast'])
    
    # Create confidence intervals for the price forecast
    price_upper_bound = [None] * len(recent_price_data) + [
        forecast_data['price_forecast'].iloc[i] * (1 + (0.1 * (i+1) / 24))
        for i in range(len(forecast_data))
    ]
    
    price_lower_bound = [None] * len(recent_price_data) + [
        forecast_data['price_forecast'].iloc[i] * (1 - (0.1 * (i+1) / 24))
        for i in range(len(forecast_data))
    ]
    
    # Create the plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=combined_price,
        mode='lines',
        name='Historical Price',
        line=dict(color='green', width=3)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=combined_price_forecast,
        mode='lines',
        name='Price Forecast',
        line=dict(color='orange', width=3, dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=price_upper_bound,
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,165,0,0.2)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=combined_timestamps,
        y=price_lower_bound,
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,165,0,0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)',
        showlegend=False
    ))
    
    # Add vertical line to separate historical from forecast
    if len(historical_timestamps) > 0:
        forecast_start_timestamp = historical_timestamps[-1]
        
        if hasattr(forecast_start_timestamp, 'timestamp'):
            forecast_start_timestamp = forecast_start_timestamp.timestamp() * 1000  # Convert to milliseconds for JavaScript
        else:
            forecast_start_timestamp = datetime.timestamp(forecast_start_timestamp) * 1000
            
        fig.add_vline(
            x=forecast_start_timestamp,
            line_dash="dash",
            line_color="black",
            annotation_text="Forecast Start",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title='Price Forecast for Charlottesville',
        xaxis_title='Time',
        yaxis_title='LMP Price ($/MWh)',
        legend=dict(x=0.01, y=0.99),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast details table
    st.markdown("### Price Forecast Details")
    
    # Create a dataframe with forecast details
    price_forecast_details = forecast_data.reset_index().copy()
    
    # Rename timestamp column if it exists
    if 'timestamp' in price_forecast_details.columns:
        price_forecast_details.rename(columns={'timestamp': 'Time'}, inplace=True)
    elif 'index' in price_forecast_details.columns:
        price_forecast_details.rename(columns={'index': 'Time'}, inplace=True)
    
    # Select and rename columns
    cols_to_keep = ['Time', 'price_forecast', 'load_forecast', 'price_confidence']
    cols_to_keep = [col for col in cols_to_keep if col in price_forecast_details.columns]
    
    price_forecast_details = price_forecast_details[cols_to_keep]
    
    # Rename columns for display
    column_mapping = {
        'price_forecast': 'Price Forecast ($/MWh)',
        'load_forecast': 'Load Forecast (MW)',
        'price_confidence': 'Confidence (%)'
    }
    
    price_forecast_details = price_forecast_details.rename(columns=column_mapping)
    
    # Format the dataframe
    if 'Time' in price_forecast_details.columns:
        price_forecast_details['Time'] = pd.to_datetime(price_forecast_details['Time']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'Price Forecast ($/MWh)' in price_forecast_details.columns:
        price_forecast_details['Price Forecast ($/MWh)'] = price_forecast_details['Price Forecast ($/MWh)'].round(2)
    
    if 'Load Forecast (MW)' in price_forecast_details.columns:
        price_forecast_details['Load Forecast (MW)'] = price_forecast_details['Load Forecast (MW)'].round(1)
    
    if 'Confidence (%)' in price_forecast_details.columns:
        price_forecast_details['Confidence (%)'] = price_forecast_details['Confidence (%)'].round(1)
    
    st.dataframe(price_forecast_details, use_container_width=True)
    
    # Price forecast drivers
    st.markdown("### Price Forecast Drivers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Load vs Price Relationship")
        
        # Reset index if it's a DatetimeIndex
        plot_data = forecast_data.reset_index() if isinstance(forecast_data.index, pd.DatetimeIndex) else forecast_data.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in plot_data.columns and 'index' in plot_data.columns:
            plot_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        if 'load_forecast' in plot_data.columns and 'price_forecast' in plot_data.columns:
            fig = px.scatter(
                plot_data,
                x='load_forecast',
                y='price_forecast',
                color='timestamp',
                labels={
                    'load_forecast': 'Load Forecast (MW)',
                    'price_forecast': 'Price Forecast ($/MWh)',
                    'timestamp': 'Hour of Day'
                },
                title='Load vs. Price Forecast',
                trendline='ols'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns not found for Load vs Price visualization")
    
    with col2:
        st.markdown("#### Hourly Price Pattern")
        
        # Reset index if it's a DatetimeIndex
        plot_data = forecast_data.reset_index() if isinstance(forecast_data.index, pd.DatetimeIndex) else forecast_data.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in plot_data.columns and 'index' in plot_data.columns:
            plot_data.rename(columns={'index': 'timestamp'}, inplace=True)
        
        if 'timestamp' in plot_data.columns and 'price_forecast' in plot_data.columns:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(plot_data['timestamp']):
                plot_data['timestamp'] = pd.to_datetime(plot_data['timestamp'])
            
            hourly_price_forecast = plot_data.groupby(plot_data['timestamp'].dt.hour)['price_forecast'].mean().reset_index()
            hourly_price_forecast.columns = ['Hour', 'Average Price Forecast ($/MWh)']
            
            fig = px.bar(
                hourly_price_forecast,
                x='Hour',
                y='Average Price Forecast ($/MWh)',
                title='Hourly Price Forecast Pattern',
                labels={
                    'Hour': 'Hour of Day',
                    'Average Price Forecast ($/MWh)': 'Average Price ($/MWh)'
                }
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns not found for Hourly Price Pattern visualization")