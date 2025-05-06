import plotly.graph_objects as go
import plotly.express as px

def create_duration_curve(data, value_column, title, x_label, y_label):
    """
    Create a duration curve for the specified data column.
    
    Args:
        data: DataFrame containing the data
        value_column: Column name for the values to plot
        title: Title for the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        
    Returns:
        Plotly figure object
    """
    sorted_values = data[value_column].sort_values(ascending=False).reset_index(drop=True)

    fig = px.bar(
        x=range(len(sorted_values)),
        y=sorted_values,
        labels={'x': x_label, 'y': y_label},
        title=title
    )

    # Add annotations for highest and lowest values
    fig.add_annotation(
        x=0, y=sorted_values.iloc[0],
        text=f"Highest: {sorted_values.iloc[0]:.2f}",
        showarrow=True, arrowhead=2
    )
    fig.add_annotation(
        x=len(sorted_values)-1, y=sorted_values.iloc[-1],
        text=f"Lowest: {sorted_values.iloc[-1]:.2f}",
        showarrow=True, arrowhead=2
    )

    # Update layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        title=title
    )
    
    return fig

def create_hourly_pattern_plot(hourly_data, value_columns, title, y_label):
    """
    Create a plot showing hourly patterns for multiple metrics.
    
    Args:
        hourly_data: DataFrame with hourly data
        value_columns: List of column names to plot
        title: Title for the plot
        y_label: Label for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for column in value_columns:
        fig.add_trace(go.Scatter(
            x=hourly_data['Hour'],
            y=hourly_data[column],
            mode='lines',
            name=column.capitalize()
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title=y_label)
    )
    
    return fig