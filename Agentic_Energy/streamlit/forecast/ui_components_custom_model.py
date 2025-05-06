import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from data.data_utils import preprocess_data_for_forecasting
from forecast.forecasting_models import evaluate_model_performance, get_feature_importance, generate_forecasts

# Function to display custom model builder tab
def display_custom_model_builder(historical_data):
    st.subheader("Custom Model Builder")
    
    # Model type selection
    model_target = st.selectbox(
        "Select Forecast Target",
        ["Load (MW)", "Price ($/MWh)"]
    )
    
    # Model algorithm selection
    model_algorithm = st.selectbox(
        "Select Algorithm",
        ["Linear Regression", "Random Forest", "Neural Network"]
    )
            
    # Input features selection
    st.markdown("### Select Input Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_temperature = st.checkbox("Temperature", value=True)
        use_humidity = st.checkbox("Humidity", value=False)
        use_wind_speed = st.checkbox("Wind Speed", value=False)

    with col2:
        use_solar_radiation = st.checkbox("Solar Radiation", value=False)
        use_hour_of_day = st.checkbox("Hour of Day", value=True)
        use_day_of_week = st.checkbox("Day of Week", value=True)

    with col3:
        use_is_weekend = st.checkbox("Is Weekend", value=False)

    # Training parameters
    st.markdown("### Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        training_days = st.slider("Training Data (days)", 500, 1825, 50)
    
    with col2:
        if model_algorithm == "Random Forest":
            n_trees = st.slider("Number of Trees", 10, 500, 100)
        elif model_algorithm == "Neural Network":
            n_layers = st.slider("Number of Hidden Layers", 1, 5, 2)
    
    # Create two columns for the buttons
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        build_model = st.button("Build and Train Model")
    
    # Track if a model was just trained in this session
    model_just_trained = False
    
    if build_model:
        with st.spinner("Training custom model..."):
            df = preprocess_data_for_forecasting(historical_data)
            
            if training_days < len(df) / 24:
                df = df.tail(training_days * 24)
            
            if model_target == "Load (MW)":
                target = 'load_mw'
            else:
                target = 'lmp_price'
            
            selected_features = []
            
            if use_temperature:
                selected_features.append('temperature')
            
            if use_humidity:
                selected_features.append('humidity')
            
            if use_wind_speed:
                selected_features.append('wind_speed')
            
            if use_solar_radiation:
                selected_features.append('solar_radiation')
            
            if use_hour_of_day:
                selected_features.extend(['hour_sin', 'hour_cos'])
            
            if use_day_of_week:
                selected_features.append('day_of_week')
            
            if use_is_weekend:
                selected_features.append('is_weekend')
            
            if not selected_features:
                st.error("Please select at least one feature for the model.")
                return
            
            X = df[selected_features]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            if model_algorithm == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
            elif model_algorithm == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
                model.fit(X_train, y_train)
            elif model_algorithm == "Neural Network":
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                
                layers = []
                layers.append(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
                
                for _ in range(n_layers - 1):
                    layers.append(Dense(32, activation='relu'))
                
                layers.append(Dense(1))
                
                model = Sequential(layers)
                
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0)
                
                model.scaler_X = scaler_X
                model.scaler_y = scaler_y
            
            metrics, performance_df = evaluate_model_performance(
                model, X_test, y_test, 
                model_type="load" if model_target == "Load (MW)" else "price"
            )
            
            # Store the model and set a flag for which type was trained
            if model_target == "Load (MW)":
                st.session_state.custom_load_model = model
                st.session_state.custom_load_features = selected_features
                st.session_state.custom_load_metrics = metrics
                st.session_state.custom_load_performance = performance_df
                st.session_state.last_trained_model_type = "load"
            else:
                st.session_state.custom_price_model = model
                st.session_state.custom_price_features = selected_features
                st.session_state.custom_price_metrics = metrics
                st.session_state.custom_price_performance = performance_df
                st.session_state.last_trained_model_type = "price"
            
            # Set flag that a model was just trained
            model_just_trained = True
            
            st.success("Model trained successfully!")
            
            st.markdown("### Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if model_target == "Load (MW)":
                    st.metric("MAE", f"{metrics['mae']:.2f} MW")
                else:
                    st.metric("MAE", f"${metrics['mae']:.2f}/MWh")
            
            with col2:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            with col3:
                st.metric("R²", f"{metrics['r2']:.3f}")
            
            if model_algorithm in ["Linear Regression", "Random Forest"]:
                st.markdown("### Feature Importance")
                
                importance = get_feature_importance(model, selected_features)
                
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                })
                
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title=f'Feature Importance for {model_target} Forecast'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Prediction vs Actual")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_df.index,
                y=performance_df['actual'],
                mode='lines',
                name='Actual'
            ))
            
            fig.add_trace(go.Scatter(
                x=performance_df.index,
                y=performance_df['predicted'],
                mode='lines',
                name='Predicted'
            ))
                
            fig.update_layout(
                title=f'{model_target} - Actual vs Predicted',
                xaxis_title='Sample',
                yaxis_title=model_target,
                legend=dict(x=0.01, y=0.99)
            )
                
            st.plotly_chart(fig, use_container_width=True)
    
    # Only show the "Apply Model" button if a model is available
    if model_just_trained or 'custom_load_model' in st.session_state or 'custom_price_model' in st.session_state:
        st.markdown("### Apply Model to Forecasts")
        
        # If a model was just trained, use that one
        if model_just_trained:
            model_type_to_apply = st.session_state.last_trained_model_type
            model_name = "Load" if model_type_to_apply == "load" else "Price"
            st.write(f"Apply the custom {model_name} model you just trained to generate new forecasts.")
        # Otherwise, let user choose which model to apply if multiple are available
        elif 'custom_load_model' in st.session_state and 'custom_price_model' in st.session_state:
            model_type_to_apply = st.radio(
                "Select which custom model to apply:",
                ["load", "price"],
                format_func=lambda x: "Load Model" if x == "load" else "Price Model"
            )
        # If only one type is available, use that
        elif 'custom_load_model' in st.session_state:
            model_type_to_apply = "load"
            st.write("Apply the custom Load model to generate new forecasts.")
        else:
            model_type_to_apply = "price"
            st.write("Apply the custom Price model to generate new forecasts.")
        
        # In display_custom_model_builder function
        with button_col2:
            if st.button("Apply Model & Update Forecasts", key="apply_model"):
                with st.spinner("Updating forecasts with custom model..."):
                    # Create a forecast period
                    last_timestamp = historical_data['timestamp'].max()
                    forecast_start = pd.to_datetime(last_timestamp) + pd.Timedelta(hours=1)
                    forecast_end = forecast_start + pd.Timedelta(hours=72)
                    
                    # Create a DataFrame with hourly timestamps
                    forecast_timestamps = pd.date_range(start=forecast_start, end=forecast_end, freq='H')
                    forecast_df = pd.DataFrame({'timestamp': forecast_timestamps})
                    
                    # Add basic features
                    forecast_df['hour'] = forecast_df['timestamp'].dt.hour
                    forecast_df['day_of_week'] = forecast_df['timestamp'].dt.dayofweek
                    forecast_df['is_weekend'] = forecast_df['day_of_week'].isin([5, 6]).astype(int)
                    
                    # Add weather features
                    forecast_df['temperature'] = 70 + 10 * np.sin(2 * np.pi * (forecast_df['hour'] - 12) / 24)
                    forecast_df['humidity'] = 50 + 10 * np.cos(2 * np.pi * forecast_df['hour'] / 24)
                    forecast_df['wind_speed'] = 5 + 3 * np.random.rand(len(forecast_df))
                    forecast_df['solar_radiation'] = np.maximum(0, 500 * np.sin(np.pi * (forecast_df['hour'] - 6) / 12))
                    forecast_df.loc[forecast_df['hour'] < 6, 'solar_radiation'] = 0
                    forecast_df.loc[forecast_df['hour'] > 18, 'solar_radiation'] = 0
                    
                    # Add cyclical time features
                    forecast_df['hour_sin'] = np.sin(2 * np.pi * forecast_df['hour'] / 24)
                    forecast_df['hour_cos'] = np.cos(2 * np.pi * forecast_df['hour'] / 24)
                    
                    # Determine which models to use
                    if model_type_to_apply == "load":
                        load_model = st.session_state.custom_load_model
                        load_features = st.session_state.custom_load_features
                        price_model = st.session_state.get('price_model', None)
                        price_features = st.session_state.get('price_features', [])
                    else:
                        load_model = st.session_state.get('load_model', None)
                        load_features = st.session_state.get('load_features', [])
                        price_model = st.session_state.custom_price_model
                        price_features = st.session_state.custom_price_features
                    
                    # Generate forecasts using the selected models
                    st.session_state.forecast_data = generate_forecasts(
                        forecast_df,
                        load_model,
                        price_model,
                        load_features,
                        price_features
                    )
                    
                    # Set flags to indicate which custom models are in use
                    if model_type_to_apply == "load":
                        st.session_state.use_custom_load_model = True
                    else:
                        st.session_state.use_custom_price_model = True
                    
                    # Set the update flag
                    st.session_state.update_forecasts = True
                    
                    # Force Streamlit to rerun the app
                    st.rerun()