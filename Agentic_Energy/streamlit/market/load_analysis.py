import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_load_analysis(filtered_data):
    st.subheader("Load Analysis")
    
    if filtered_data.empty:
        st.warning("No data available for selected date range.")
        return
    
    # Load statistics
    st.markdown("### Load Statistics")
    load_stats = filtered_data['load_mw'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Load", f"{load_stats['mean']:.1f} MW")
    with col2:
        st.metric("Peak Load", f"{load_stats['max']:.1f} MW")
    with col3:
        st.metric("Minimum Load", f"{load_stats['min']:.1f} MW")
    with col4:
        load_factor = load_stats['mean'] / load_stats['max'] * 100 if load_stats['max'] > 0 else 0
        st.metric("Load Factor", f"{load_factor:.1f}%")
    
    # Load duration curve with bar chart
    st.markdown("### Load Duration Curve")
    sorted_load = filtered_data['load_mw'].sort_values(ascending=False).reset_index(drop=True)

    fig = px.bar(
        x=range(len(sorted_load)),
        y=sorted_load,
        labels={'x': 'Data Points (sorted by load)', 'y': 'Load (MW)'},
        title='Load Duration Curve'
    )


    # Update layout for better understanding
    fig.update_layout(
        xaxis_title='Data Points (sorted by load)',
        yaxis_title='Load (MW)',
        title='Load Duration Curve (Sorted Loads)',
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly load patterns
    st.markdown("### Hourly Load Patterns")
    
    # Group by hour of day
    hourly_load = filtered_data.groupby(filtered_data['timestamp'].dt.hour)['load_mw'].agg(['mean', 'min', 'max'])
    hourly_load.reset_index(inplace=True)
    hourly_load.rename(columns={'timestamp': 'Hour'}, inplace=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_load['Hour'],
        y=hourly_load['mean'],
        mode='lines',
        name='Average Load',
        line=dict(color='blue', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=hourly_load['Hour'],
        y=hourly_load['max'],
        mode='lines',
        name='Maximum Load',
        line=dict(color='red', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=hourly_load['Hour'],
        y=hourly_load['min'],
        mode='lines',
        name='Minimum Load',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Hourly Load Patterns',
        xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title='Load (MW)'),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature vs Load scatter plot
    st.markdown("### Temperature vs Load Relationship")
    
    fig = px.scatter(
        filtered_data,
        x='temperature',
        y='load_mw',
        color='timestamp',
        title='Temperature vs Load',
        labels={'temperature': 'Temperature (°F)', 'load_mw': 'Load (MW)', 'timestamp': 'Hour of Day'},
        trendline='ols'
    )
    
    st.plotly_chart(fig, use_container_width=True)