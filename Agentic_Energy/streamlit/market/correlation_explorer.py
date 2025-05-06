import streamlit as st
import plotly.express as px
import pandas as pd

def render_correlation_explorer(filtered_data):
    st.subheader("Correlation Explorer")
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Select variables for correlation analysis
    st.markdown("### Select Variables for Correlation Analysis")
    
    available_vars = [
        'load_mw', 'lmp_price', 'temperature', 'humidity', 'wind_speed', 
        'solar_radiation', 'solar_generation', 'wind_generation', 
        'natural_gas_generation', 'other_generation', 'congestion_index'
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "X-axis Variable",
            available_vars,
            index=available_vars.index('temperature')
        )
    
    with col2:
        y_var = st.selectbox(
            "Y-axis Variable",
            available_vars,
            index=available_vars.index('load_mw')
        )
    
    # Create scatter plot
    st.markdown("### Correlation Scatter Plot")
    
    fig = px.scatter(
        filtered_data,
        x=x_var,
        y=y_var,
        color='timestamp',
        title=f'{x_var.capitalize()} vs {y_var.capitalize()}',
        labels={
            x_var: x_var.replace('_', ' ').capitalize(),
            y_var: y_var.replace('_', ' ').capitalize(),
            'timestamp': 'Hour of Day'
        },
        trendline='ols'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### Correlation Matrix")
    
    # Select variables for correlation matrix
    correlation_vars = st.multiselect(
        "Select Variables for Correlation Matrix",
        available_vars,
        default=['load_mw', 'lmp_price', 'temperature', 'congestion_index']
    )
    
    if correlation_vars:
        corr_matrix = filtered_data[correlation_vars].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix',
            labels=dict(x="Variable", y="Variable", color="Correlation")
        )
        
        fig.update_layout(
            height=500,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation statistics
        st.markdown("### Correlation Statistics")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))