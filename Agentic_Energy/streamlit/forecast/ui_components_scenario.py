import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to display scenario analysis tab
def display_scenario_analysis(forecast_data):
    st.subheader("Scenario Analysis")
    
    st.markdown("""
    ### Forecast Scenario Analysis
    
    Explore how different scenarios might affect load and price forecasts.
    """)
    
    scenario = st.selectbox(
        "Select Scenario",
        ["Base Case", "Heat Wave", "Cold Snap", "High Wind Generation", "Low Solar Output"]
    )
    
    st.markdown("### Scenario Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if scenario == "Heat Wave" or scenario == "Cold Snap":
            temp_adjustment = st.slider(
                "Temperature Adjustment (°F)",
                -20, 20, 
                10 if scenario == "Heat Wave" else -10
            )
        
        if scenario == "High Wind Generation":
            wind_adjustment = st.slider("Wind Generation Multiplier", 1.0, 3.0, 1.5, step=0.1)
        
        if scenario == "Low Solar Output":
            solar_adjustment = st.slider("Solar Generation Multiplier", 0.1, 1.0, 0.5, step=0.1)
    
    with col2:
        scenario_days = st.slider("Scenario Duration (days)", 1, 7, 3)
    
    if st.button("Run Scenario Analysis"):
        with st.spinner("Analyzing scenario..."):
            base_forecast = forecast_data.copy()
            
            base_forecast = base_forecast.head(scenario_days * 24)
            
            scenario_params = {}
            
            if scenario == "Heat Wave" or scenario == "Cold Snap":
                scenario_params['temp_adjustment'] = temp_adjustment
            
            if scenario == "High Wind Generation":
                scenario_params['wind_adjustment'] = wind_adjustment
            
            if scenario == "Low Solar Output":
                scenario_params['solar_adjustment'] = solar_adjustment
            
            scenario_forecast = apply_scenario(base_forecast, scenario, scenario_params)
            
            st.success(f"Scenario analysis complete: {scenario}")
            
            st.markdown("### Load Forecast Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=base_forecast['timestamp'],
                y=base_forecast['load_forecast'],
                mode='lines',
                name='Base Case',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=scenario_forecast['timestamp'],
                y=scenario_forecast['load_forecast'],
                mode='lines',
                name=scenario,
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'Load Forecast: Base Case vs {scenario}',
                xaxis_title='Time',
                yaxis_title='Load (MW)',
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Price Forecast Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=base_forecast['timestamp'],
                y=base_forecast['price_forecast'],
                mode='lines',
                name='Base Case',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=scenario_forecast['timestamp'],
                y=scenario_forecast['price_forecast'],
                mode='lines',
                name=scenario,
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title=f'Price Forecast: Base Case vs {scenario}',
                xaxis_title='Time',
                yaxis_title='Price ($/MWh)',
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Scenario Impact Summary")
            
            avg_load_base = base_forecast['load_forecast'].mean()
            avg_load_scenario = scenario_forecast['load_forecast'].mean()
            load_change_pct = (avg_load_scenario / avg_load_base - 1) * 100
            
            avg_price_base = base_forecast['price_forecast'].mean()
            avg_price_scenario = scenario_forecast['price_forecast'].mean()
            price_change_pct = (avg_price_scenario / avg_price_base - 1) * 100
            
            peak_load_base = base_forecast['load_forecast'].max()
            peak_load_scenario = scenario_forecast['load_forecast'].max()
            peak_load_change_pct = (peak_load_scenario / peak_load_base - 1) * 100
            
            peak_price_base = base_forecast['price_forecast'].max()
            peak_price_scenario = scenario_forecast['price_forecast'].max()
            peak_price_change_pct = (peak_price_scenario / peak_price_base - 1) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Load Impacts")
                st.metric("Average Load Change", f"{load_change_pct:.1f}%", 
                          delta=f"{load_change_pct:.1f}%",
                          delta_color="inverse" if load_change_pct < 0 else "normal")
                st.metric("Peak Load Change", f"{peak_load_change_pct:.1f}%", 
                          delta=f"{peak_load_change_pct:.1f}%",
                          delta_color="inverse" if peak_load_change_pct < 0 else "normal")
            
            with col2:
                st.markdown("#### Price Impacts")
                st.metric("Average Price Change", f"{price_change_pct:.1f}%", 
                          delta=f"{price_change_pct:.1f}%",
                          delta_color="inverse" if price_change_pct < 0 else "normal")
                st.metric("Peak Price Change", f"{peak_price_change_pct:.1f}%", 
                          delta=f"{peak_price_change_pct:.1f}%",
                          delta_color="inverse" if peak_price_change_pct < 0 else "normal")
            
            st.markdown("### Recommendations")
            
            if scenario == "Heat Wave":
                st.info("• Consider demand response to reduce peak load during afternoon hours")
                st.info("• Monitor transmission constraints as they may become binding during peak hours")
                st.info("• Evaluate price hedging strategies for peak hours")
                st.info("• Increase cooling capacity and ensure backup systems are operational")
                st.info("• Prepare for potential supply shortages during extended heat events")
            
            elif scenario == "Cold Snap":
                st.info("• Prepare for higher morning and evening peaks")
                st.info("• Monitor natural gas supply and prices as they may impact electricity prices")
                st.info("• Consider load shifting strategies to reduce exposure to price spikes")
                st.info("• Ensure heating systems are fully operational and backup systems are ready")
                st.info("• Watch for fuel supply constraints that could affect generation availability")
            
            elif scenario == "High Wind Generation":
                st.info("• Take advantage of lower prices during high wind periods")
                st.info("• Monitor for potential negative pricing during overnight hours")
                st.info("• Be prepared for price volatility if wind generation suddenly drops")
                st.info("• Consider shifting flexible loads to periods of high wind generation")
                st.info("• Watch for transmission congestion that could limit wind deliverability")
            
            elif scenario == "Low Solar Output":
                st.info("• Expect higher prices during typical solar peak hours (10am-4pm)")
                st.info("• Consider load reduction strategies during afternoon hours")
                st.info("• Monitor for potential price spikes if low solar coincides with high demand")
                st.info("• Ensure non-solar generation resources are available to compensate")
                st.info("• Evaluate the need for additional hedging during low solar periods")
            
            st.markdown("### Hourly Impact Details")
            
            impact_df = pd.DataFrame({
                'Time': scenario_forecast['timestamp'],
                'Base Load (MW)': base_forecast['load_forecast'],
                'Scenario Load (MW)': scenario_forecast['load_forecast'],
                'Load Change (%)': ((scenario_forecast['load_forecast'] / base_forecast['load_forecast']) - 1) * 100,
                'Base Price ($/MWh)': base_forecast['price_forecast'],
                'Scenario Price ($/MWh)': scenario_forecast['price_forecast'],
                'Price Change (%)': ((scenario_forecast['price_forecast'] / base_forecast['price_forecast']) - 1) * 100
            })
            
            impact_df['Time'] = impact_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
            impact_df['Base Load (MW)'] = impact_df['Base Load (MW)'].round(1)
            impact_df['Scenario Load (MW)'] = impact_df['Scenario Load (MW)'].round(1)
            impact_df['Load Change (%)'] = impact_df['Load Change (%)'].round(1)
            impact_df['Base Price ($/MWh)'] = impact_df['Base Price ($/MWh)'].round(2)
            impact_df['Scenario Price ($/MWh)'] = impact_df['Scenario Price ($/MWh)'].round(2)
            impact_df['Price Change (%)'] = impact_df['Price Change (%)'].round(1)
            
            st.dataframe(impact_df, use_container_width=True)
            
            st.markdown("### Risk Assessment")
            
            max_load_increase = impact_df['Load Change (%)'].max()
            max_price_increase = impact_df['Price Change (%)'].max()
            
            hours_with_significant_load_impact = sum(impact_df['Load Change (%)'].abs() > 5)
            hours_with_significant_price_impact = sum(impact_df['Price Change (%)'].abs() > 10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Load Risk")
                
                if max_load_increase > 20:
                    load_risk = "High"
                    load_risk_color = "red"
                elif max_load_increase > 10:
                    load_risk = "Medium"
                    load_risk_color = "orange"
                else:
                    load_risk = "Low"
                    load_risk_color = "green"
                
                load_risk_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=max_load_increase,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Maximum Load Impact"},
                    delta={'reference': 0, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 30], 'tickwidth': 1},
                        'bar': {'color': load_risk_color},
                        'steps': [
                            {'range': [0, 10], 'color': "lightgreen"},
                            {'range': [10, 20], 'color': "orange"},
                            {'range': [20, 30], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_load_increase
                        }
                    }
                ))
                
                load_risk_gauge.update_layout(height=250)
                st.plotly_chart(load_risk_gauge, use_container_width=True)
                
                st.metric("Hours with Significant Impact", f"{hours_with_significant_load_impact} hours")
            
            with col2:
                st.markdown("#### Price Risk")
                
                if max_price_increase > 30:
                    price_risk = "High"
                    price_risk_color = "red"
                elif max_price_increase > 15:
                    price_risk = "Medium"
                    price_risk_color = "orange"
                else:
                    price_risk = "Low"
                    price_risk_color = "green"
                
                price_risk_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=max_price_increase,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Maximum Price Impact"},
                    delta={'reference': 0, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 50], 'tickwidth': 1},
                        'bar': {'color': price_risk_color},
                        'steps': [
                            {'range': [0, 15], 'color': "lightgreen"},
                            {'range': [15, 30], 'color': "orange"},
                            {'range': [30, 50], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_price_increase
                        }
                    }
                ))
                
                price_risk_gauge.update_layout(height=250)
                st.plotly_chart(price_risk_gauge, use_container_width=True)
                
                st.metric("Hours with Significant Impact", f"{hours_with_significant_price_impact} hours")
            
            st.markdown("#### Overall Risk Assessment")
            
            if load_risk == "High" or price_risk == "High":
                overall_risk = "High"
                risk_color = "red"
            elif load_risk == "Medium" or price_risk == "Medium":
                overall_risk = "Medium"
                risk_color = "orange"
            else:
                overall_risk = "Low"
                risk_color = "green"
            
            st.markdown(f"<h3 style='color:{risk_color};'>Overall Risk: {overall_risk}</h3>", unsafe_allow_html=True)
            
            st.markdown("### Risk Mitigation Strategies")
            
            if overall_risk == "High":
                st.warning("This scenario presents significant risks that require immediate attention and action.")
                
                if scenario == "Heat Wave":
                    st.markdown("""
                    1. **Activate Demand Response Programs**: Reduce peak demand during critical hours
                    2. **Secure Additional Generation**: Ensure all available generation resources are operational
                    3. **Implement Load Shedding Plan**: Prepare for potential emergency load reductions
                    4. **Increase Market Hedging**: Secure fixed-price contracts to mitigate price exposure
                    5. **Coordinate with Large Customers**: Work with major users to reduce consumption during peaks
                    """)
                
                elif scenario == "Cold Snap":
                    st.markdown("""
                    1. **Ensure Fuel Supply**: Verify adequate natural gas and other fuel supplies
                    2. **Activate Heating Demand Response**: Target heating load reductions during peak hours
                    3. **Secure Generation Reserves**: Ensure backup generation is available
                    4. **Implement Price Hedging**: Lock in prices before significant increases
                    5. **Coordinate with Gas Utilities**: Ensure coordination between gas and electric operations
                    """)
                
                elif scenario == "Low Solar Output":
                    st.markdown("""
                    1. **Secure Alternative Generation**: Ensure non-solar resources are available
                    2. **Implement Demand Response**: Target mid-day load reductions
                    3. **Increase Storage Utilization**: Deploy energy storage to offset solar reduction
                    4. **Adjust Market Positions**: Increase purchases during affected periods
                    5. **Prepare Load Shifting Strategies**: Move flexible loads to periods with higher generation
                    """)
            
            elif overall_risk == "Medium":
                st.info("This scenario presents moderate risks that should be monitored closely.")
                
                st.markdown("""
                1. **Monitor System Conditions**: Watch for changing conditions that could increase risk
                2. **Prepare Demand Response**: Have programs ready but not necessarily activated
                3. **Review Hedging Positions**: Consider adjusting market positions
                4. **Alert Operations Teams**: Ensure awareness of potential issues
                5. **Increase Forecasting Frequency**: Update forecasts more frequently to capture changes
                """)
            
            else:
                st.success("This scenario presents manageable risks that can be addressed through normal operations.")
                
                st.markdown("""
                1. **Standard Monitoring**: Maintain normal system monitoring
                2. **Regular Updates**: Continue with standard forecast updates
                3. **Normal Operations**: Proceed with planned operations
                4. **Document Lessons**: Record any insights for future reference
                """)
                
    def apply_scenario(forecast_df, scenario, params):
        scenario_df = forecast_df.copy()
        
        if scenario == "Heat Wave":
            temp_adjustment = params.get('temp_adjustment', 10)
            scenario_df['temperature'] += temp_adjustment
            
            temp_sensitivity = 2.5
            scenario_df['load_forecast'] += temp_adjustment * temp_sensitivity
            
            price_sensitivity = 0.8
            scenario_df['price_forecast'] += (temp_adjustment * temp_sensitivity) * price_sensitivity
            
        elif scenario == "Cold Snap":
            temp_adjustment = params.get('temp_adjustment', -10)
            scenario_df['temperature'] += temp_adjustment
            
            temp_sensitivity = 3.0
            scenario_df['load_forecast'] += abs(temp_adjustment) * temp_sensitivity
            
            price_sensitivity = 1.0
            scenario_df['price_forecast'] += (abs(temp_adjustment) * temp_sensitivity) * price_sensitivity
            
        elif scenario == "High Wind Generation":
            wind_adjustment = params.get('wind_adjustment', 1.5)
            scenario_df['wind_forecast'] *= wind_adjustment
            
            additional_wind = (wind_adjustment - 1) * scenario_df['wind_forecast']
            price_reduction = additional_wind * 0.5
            scenario_df['price_forecast'] -= price_reduction
            
            scenario_df['price_forecast'] = scenario_df['price_forecast'].clip(lower=10)
            
        elif scenario == "Low Solar Output":
            solar_adjustment = params.get('solar_adjustment', 0.5)
            scenario_df['solar_forecast'] *= solar_adjustment
            
            solar_reduction = (1 - solar_adjustment) * forecast_df['solar_forecast']
            price_increase = solar_reduction * 0.7
            scenario_df['price_forecast'] += price_increase
        
        return scenario_df