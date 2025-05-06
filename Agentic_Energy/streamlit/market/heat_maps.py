import streamlit as st
import plotly.express as px
import pandas as pd

def render_heat_maps(filtered_data):
    st.subheader("Heat Maps")
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Heat map options
    heat_map_var = st.selectbox(
        "Select Variable for Heat Map",
        ['lmp_price', 'load_mw', 'congestion_index', 'temperature', 'humidity', 'wind_speed', 'solar_radiation'],
        index=0
    )
    
    # Create day-hour heatmap
    st.markdown("### Day-Hour Heat Map")
    
    # Ensure timestamp is in datetime format
    filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'])
    
    # Extract day and hour
    filtered_data['day'] = filtered_data['timestamp'].dt.date
    filtered_data['hour'] = filtered_data['timestamp'].dt.hour
    
    # Pivot data for heatmap
    pivot_data = filtered_data.pivot_table(
        index='day',
        columns='hour',
        values=heat_map_var,
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Date", color=heat_map_var.replace('_', ' ').capitalize()),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='Viridis',
        title=f'{heat_map_var.replace("_", " ").capitalize()} Heat Map'
    )
    
    fig.update_layout(
        height=600,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week - hour heatmap
    st.markdown("### Day of Week - Hour Heat Map")
    
    # Use the provided day_of_week column
    filtered_data['day_name'] = filtered_data['day_of_week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    # Pivot data for heatmap
    dow_pivot = filtered_data.pivot_table(
        index='day_name',
        columns='hour',
        values=heat_map_var,
        aggfunc='mean'
    )
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_pivot = dow_pivot.reindex(day_order)
    
    # Create heatmap
    fig = px.imshow(
        dow_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color=heat_map_var.replace('_', ' ').capitalize()),
        x=dow_pivot.columns,
        y=dow_pivot.index,
        color_continuous_scale='Viridis',
        title=f'Average {heat_map_var.replace("_", " ").capitalize()} by Day of Week and Hour'
    )
    
    fig.update_layout(
        height=400,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)