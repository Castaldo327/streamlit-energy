import streamlit as st
import plotly.express as px
import pandas as pd

def render_generation_mix(filtered_data, start_date, end_date):
    st.subheader("Generation Mix Analysis")
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Calculate total generation by source
    total_solar = filtered_data['solar_generation'].sum()
    total_wind = filtered_data['wind_generation'].sum()
    total_natural_gas = filtered_data['natural_gas_generation'].sum()
    total_other = filtered_data['other_generation'].sum()
    total_generation = total_solar + total_wind + total_natural_gas + total_other
    
    # Generation mix pie chart
    st.markdown("### Overall Generation Mix")
    
    generation_mix = pd.DataFrame({
        'Source': ['Solar', 'Wind', 'Natural Gas', 'Other'],
        'Generation (MWh)': [total_solar, total_wind, total_natural_gas, total_other],
        'Percentage': [
            total_solar / total_generation * 100 if total_generation > 0 else 0,
            total_wind / total_generation * 100 if total_generation > 0 else 0,
            total_natural_gas / total_generation * 100 if total_generation > 0 else 0,
            total_other / total_generation * 100 if total_generation > 0 else 0
        ]
    })
    
    fig = px.pie(
        generation_mix,
        values='Generation (MWh)',
        names='Source',
        title='Generation Mix',
        color='Source',
        color_discrete_map={
            'Solar': 'gold',
            'Wind': 'skyblue',
            'Natural Gas': 'gray',
            'Other': 'darkgreen'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generation mix over time
    st.markdown("### Generation Mix Over Time")
    
    # Resample to daily for better visualization if date range is long
    if (end_date - start_date).days > 7:
        daily_generation = filtered_data.set_index('timestamp').resample('D').sum().reset_index()
        
        fig = px.area(
            daily_generation,
            x='timestamp',
            y=['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation'],
            title='Daily Generation Mix',
            labels={
                'timestamp': 'Date',
                'value': 'Generation (MWh)',
                'variable': 'Source'
            },
            color_discrete_map={
                'solar_generation': 'gold',
                'wind_generation': 'skyblue',
                'natural_gas_generation': 'gray',
                'other_generation': 'darkgreen'
            }
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Generation (MWh)',
            legend_title='Source',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Use hourly data for shorter periods
        fig = px.area(
            filtered_data,
            x='timestamp',
            y=['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation'],
            title='Hourly Generation Mix',
            labels={
                'timestamp': 'Time',
                'value': 'Generation (MW)',
                'variable': 'Source'
            },
            color_discrete_map={
                'solar_generation': 'gold',
                'wind_generation': 'skyblue',
                'natural_gas_generation': 'gray',
                'other_generation': 'darkgreen'
            }
        )
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Generation (MW)',
            legend_title='Source',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Average hourly generation by source
    st.markdown("### Average Hourly Generation by Source")
    
    hourly_generation = filtered_data.groupby(filtered_data['timestamp'].dt.hour).agg({
        'solar_generation': 'mean',
        'wind_generation': 'mean',
        'natural_gas_generation': 'mean',
        'other_generation': 'mean'
    }).reset_index()
    
    fig = px.line(
        hourly_generation,
        x='timestamp',
        y=['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation'],
        title='Average Hourly Generation by Source',
        labels={
            'timestamp': 'Hour of Day',
            'value': 'Average Generation (MW)',
            'variable': 'Source'
        },
        color_discrete_map={
            'solar_generation': 'gold',
            'wind_generation': 'skyblue',
            'natural_gas_generation': 'gray',
            'other_generation': 'darkgreen'
        }
    )
    
    fig.update_layout(
        xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
        yaxis_title='Average Generation (MW)',
        legend_title='Source'
    )
    
    st.plotly_chart(fig, use_container_width=True)