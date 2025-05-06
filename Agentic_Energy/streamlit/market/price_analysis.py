import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_price_analysis(filtered_data):
    st.subheader("Price Analysis")
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Price statistics
    st.markdown("### Price Statistics")
    price_stats = filtered_data['lmp_price'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Price", f"${price_stats['mean']:.2f}")
    with col2:
        st.metric("Maximum Price", f"${price_stats['max']:.2f}")
    with col3:
        st.metric("Minimum Price", f"${price_stats['min']:.2f}")
    with col4:
        st.metric("Price Volatility", f"{price_stats['std']:.2f}")
    
    # Price duration curve with bar chart
    st.markdown("### Price Duration Curve")
    sorted_prices = filtered_data['lmp_price'].sort_values(ascending=False).reset_index(drop=True)

    fig = px.bar(
        x=range(len(sorted_prices)),
        y=sorted_prices,
        labels={'x': 'Data Points (sorted by price)', 'y': 'Price ($/MWh)'},
        title='Price Duration Curve'
    )
    
    # Update layout for better understanding
    fig.update_layout(
        xaxis_title='Data Points (sorted by price)',
        yaxis_title='Price ($/MWh)',
        title='Price Duration Curve (Sorted Prices)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly price patterns
    st.markdown("### Hourly Price Patterns")
    
    # Group by hour of day
    hourly_prices = filtered_data.groupby(filtered_data['timestamp'].dt.hour)['lmp_price'].agg(['mean', 'min', 'max'])
    hourly_prices.reset_index(inplace=True)
    hourly_prices.rename(columns={'timestamp': 'Hour'}, inplace=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_prices['Hour'],
        y=hourly_prices['mean'],
        mode='lines',
        name='Average Price',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=hourly_prices['Hour'],
        y=hourly_prices['max'],
        mode='lines',
        name='Maximum Price',
        line=dict(color='red', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=hourly_prices['Hour'],
        y=hourly_prices['min'],
        mode='lines',
        name='Minimum Price',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Hourly Price Patterns',
        xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title='Price ($/MWh)'),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week patterns
    st.markdown("### Day of Week Patterns")
    
    # Group by day of week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_prices = filtered_data.groupby(filtered_data['day_of_week'])['lmp_price'].mean().reset_index()
    daily_prices['day_name'] = daily_prices['day_of_week'].apply(lambda x: day_names[x])
    
    fig = px.bar(
        daily_prices,
        x='day_name',
        y='lmp_price',
        title='Average Price by Day of Week',
        labels={'lmp_price': 'Average Price ($/MWh)', 'day_name': 'Day of Week'},
        category_orders={"day_name": day_names}
    )
    
    st.plotly_chart(fig, use_container_width=True)