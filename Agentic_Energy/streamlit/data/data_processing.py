import pandas as pd
import numpy as np

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to calculate statistics for
        
    Returns:
        Dictionary of statistics
    """
    stats = data[column].describe()
    return {
        'mean': stats['mean'],
        'max': stats['max'],
        'min': stats['min'],
        'std': stats['std'],
        'median': stats['50%']
    }

def aggregate_by_hour(data, value_columns, agg_functions=None):
    """
    Aggregate data by hour of day.
    
    Args:
        data: DataFrame containing timestamp column
        value_columns: List of columns to aggregate
        agg_functions: Dictionary of aggregation functions for each column
        
    Returns:
        DataFrame with hourly aggregated data
    """
    if agg_functions is None:
        agg_functions = {col: ['mean', 'min', 'max'] for col in value_columns}
    
    hourly_data = data.groupby(data['timestamp'].dt.hour).agg(agg_functions)
    hourly_data.reset_index(inplace=True)
    hourly_data.rename(columns={'timestamp': 'Hour'}, inplace=True)
    
    return hourly_data

def aggregate_by_day_of_week(data, value_column):
    """
    Aggregate data by day of week.
    
    Args:
        data: DataFrame containing day_of_week column
        value_column: Column to aggregate
        
    Returns:
        DataFrame with data aggregated by day of week
    """
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_data = data.groupby(data['day_of_week'])[value_column].mean().reset_index()
    daily_data['day_name'] = daily_data['day_of_week'].apply(lambda x: day_names[x])
    
    return daily_data

def create_pivot_heatmap_data(data, index_col, columns_col, values_col, agg_func='mean'):
    """
    Create pivot table data for heatmaps.
    
    Args:
        data: DataFrame containing the data
        index_col: Column to use as index
        columns_col: Column to use as columns
        values_col: Column to use as values
        agg_func: Aggregation function to use
        
    Returns:
        Pivot table DataFrame
    """
    pivot_data = data.pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc=agg_func
    )
    
    return pivot_data