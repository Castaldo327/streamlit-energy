import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split

from data.data_utils import preprocess_data_for_forecasting
from forecast.forecasting_models import evaluate_model_performance, get_feature_importance

# Function to display model performance tab
def display_model_performance(historical_data, load_model, price_model, load_features, price_features):
    st.subheader("Forecast Model Performance")
    
    # Select model type
    model_type = st.radio("Select Model Type", ["Load Forecast", "Price Forecast"])
    
    # Get model performance data
    if model_type == "Load Forecast":
        df = preprocess_data_for_forecasting(historical_data)
        X = df[load_features]
        y = df['load_mw']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        metrics, performance_df = evaluate_model_performance(
            load_model, X_test, y_test, model_type="load"
        )
        
        st.markdown("### Load Forecast Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average MAE", f"{metrics['mae']:.2f} MW")
        
        with col2:
            st.metric("Average MAPE", f"{metrics['mape']:.2f}%")
        
        with col3:
            st.metric("Average RMSE", f"{metrics['rmse']:.2f} MW")
        
        st.markdown("### Performance Over Time")
        
        performance_df['timestamp'] = df.iloc[-len(performance_df):]['timestamp'].values
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_df['timestamp'],
            y=performance_df['actual'],
            mode='lines',
            name='Actual Load'
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_df['timestamp'],
            y=performance_df['predicted'],
            mode='lines',
            name='Predicted Load'
        ))
        
        fig.update_layout(
            title='Load Forecast vs Actual Values',
            xaxis_title='Date',
            yaxis_title='Load (MW)',
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Error Distribution")
        
        fig = px.histogram(
            performance_df,
            x='error',
            nbins=30,
            title='Load Forecast Error Distribution',
            labels={'error': 'Forecast Error (MW)', 'count': 'Frequency'}
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Feature Importance")
        
        importance = get_feature_importance(load_model, load_features)
        
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Feature',
            y='Importance',
            title='Feature Importance for Load Forecast'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Price Forecast
        df = preprocess_data_for_forecasting(historical_data)
        X = df[price_features]
        y = df['lmp_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        metrics, performance_df = evaluate_model_performance(
            price_model, X_test, y_test, model_type="price"
        )
        
        st.markdown("### Price Forecast Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average MAE", f"${metrics['mae']:.2f}/MWh")
        
        with col2:
            st.metric("Average MAPE", f"{metrics['mape']:.2f}%")
        
        with col3:
            st.metric("Average RMSE", f"${metrics['rmse']:.2f}/MWh")
        
        st.markdown("### Performance Over Time")
        
        performance_df['timestamp'] = df.iloc[-len(performance_df):]['timestamp'].values
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_df['timestamp'],
            y=performance_df['actual'],
            mode='lines',
            name='Actual Price'
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_df['timestamp'],
            y=performance_df['predicted'],
            mode='lines',
            name='Predicted Price'
        ))
        
        fig.update_layout(
            title='Price Forecast vs Actual Values',
            xaxis_title='Date',
            yaxis_title='Price ($/MWh)',
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Error Distribution")
        
        fig = px.histogram(
            performance_df,
            x='error',
            nbins=30,
            title='Price Forecast Error Distribution',
            labels={'error': 'Forecast Error ($/MWh)', 'count': 'Frequency'}
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Feature Importance")
        
        importance = get_feature_importance(price_model, price_features)
        
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Feature',
            y='Importance',
            title='Feature Importance for Price Forecast'
        )
        
        st.plotly_chart(fig, use_container_width=True)