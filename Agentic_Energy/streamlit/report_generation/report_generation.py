import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64


def report_generation():
    st.subheader("Generate Custom Reports for Charlottesville Energy Market")
    
    st.markdown("""
    ### Custom Report Builder
    
    Create customized reports for different stakeholders with the data and insights you need.
    """)
    
    # Report configuration
    st.markdown("### Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", "Charlottesville Energy Market Analysis")
        report_author = st.text_input("Author", "Power Analyst")
        
        report_type = st.selectbox(
            "Report Type",
            ["Daily Brief", "Weekly Summary", "Monthly Analysis", "Custom"]
        )
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=6 if report_type == "Weekly Summary" else 1)
        )
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date()
        )
        
        include_forecast = st.checkbox("Include Forecast", value=True)
        if include_forecast:
            forecast_days = st.slider("Forecast Days", 1, 7, 3)
    
    # Report sections selection
    st.markdown("### Report Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_executive_summary = st.checkbox("Executive Summary", value=True)
        include_market_prices = st.checkbox("Market Prices", value=True)
        include_load_analysis = st.checkbox("Load Analysis", value=True)
        include_generation_mix = st.checkbox("Generation Mix", value=True)
    
    with col2:
        include_weather_impact = st.checkbox("Weather Impact", value=True)
        include_forecast_section = st.checkbox("Forecasts", value=True)
        include_recommendations = st.checkbox("Recommendations", value=True)
        include_raw_data = st.checkbox("Raw Data Tables", value=False)
    
    # Generate report button
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Filter data for report period
            mask = (st.session_state.cville_data['timestamp'].dt.date >= start_date) & \
                   (st.session_state.cville_data['timestamp'].dt.date <= end_date)
            report_data = st.session_state.cville_data[mask]
            
            if report_data.empty:
                st.error("No data available for the selected date range.")
            else:
                # Create report container
                report_container = st.container()
                
                with report_container:
                    # Report header
                    st.markdown(f"# {report_title}")
                    st.markdown(f"**Author:** {report_author}")
                    st.markdown(f"**Period:** {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")
                    st.markdown(f"**Generated:** {datetime.now().strftime('%b %d, %Y %H:%M')}")
                    st.markdown("---")
                    
                    # Executive Summary
                    if include_executive_summary:
                        st.markdown("## Executive Summary")
                        
                        # Calculate key metrics for summary
                        avg_price = report_data['lmp_price'].mean()
                        max_price = report_data['lmp_price'].max()
                        avg_load = report_data['load_mw'].mean()
                        peak_load = report_data['load_mw'].max()
                        renewable_pct = (report_data['solar_generation'].sum() + report_data['wind_generation'].sum()) / \
                                        report_data['load_mw'].sum() * 100 if report_data['load_mw'].sum() > 0 else 0
                        
                        # Create summary text
                        summary_text = f"""
                        During the period from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}, 
                        the Charlottesville energy market experienced an average price of **${avg_price:.2f}/MWh** with 
                        a peak price of **${max_price:.2f}/MWh**. The average load was **{avg_load:.1f} MW** with a 
                        peak load of **{peak_load:.1f} MW**. Renewable energy sources (solar and wind) accounted for 
                        **{renewable_pct:.1f}%** of total energy consumption during this period.
                        """
                        
                        st.markdown(summary_text)
                        
                        # Add key metrics as a dashboard
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Price", f"${avg_price:.2f}/MWh")
                        
                        with col2:
                            st.metric("Peak Price", f"${max_price:.2f}/MWh")
                        
                        with col3:
                            st.metric("Avg Load", f"{avg_load:.1f} MW")
                        
                        with col4:
                            st.metric("Peak Load", f"{peak_load:.1f} MW")
                    
                    # Market Prices
                    if include_market_prices:
                        st.markdown("## Market Prices")
                        
                        # Price trend chart
                        fig = px.line(
                            report_data,
                            x='timestamp',
                            y='lmp_price',
                            title='LMP Price Trend',
                            labels={'lmp_price': 'Price ($/MWh)', 'timestamp': 'Time'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Price statistics
                        st.markdown("### Price Statistics")
                        
                        price_stats = report_data['lmp_price'].describe().round(2)
                        st.dataframe(price_stats)
                        
                        # Price duration curve
                        st.markdown("### Price Duration Curve")
                        
                        sorted_prices = report_data['lmp_price'].sort_values(ascending=False).reset_index(drop=True)
                        
                        fig = px.line(
                            x=range(len(sorted_prices)),
                            y=sorted_prices,
                            labels={'x': 'Hours', 'y': 'Price ($/MWh)'},
                            title='Price Duration Curve'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Load Analysis
                    if include_load_analysis:
                        st.markdown("## Load Analysis")
                        
                        # Load trend chart
                        fig = px.line(
                            report_data,
                            x='timestamp',
                            y='load_mw',
                            title='Load Trend',
                            labels={'load_mw': 'Load (MW)', 'timestamp': 'Time'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Load statistics
                        st.markdown("### Load Statistics")
                        
                        load_stats = report_data['load_mw'].describe().round(1)
                        st.dataframe(load_stats)
                        
                        # Load profile by hour of day
                        st.markdown("### Average Load Profile by Hour of Day")
                        
                        hourly_load = report_data.groupby(report_data['timestamp'].dt.hour)['load_mw'].mean().reset_index()
                        hourly_load.columns = ['Hour', 'Average Load (MW)']
                        
                        fig = px.bar(
                            hourly_load,
                            x='Hour',
                            y='Average Load (MW)',
                            title='Average Hourly Load Profile',
                            labels={'Hour': 'Hour of Day', 'Average Load (MW)': 'Load (MW)'}
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Generation Mix
                    if include_generation_mix:
                        st.markdown("## Generation Mix")
                        
                        # Calculate total generation by source
                        total_solar = report_data['solar_generation'].sum()
                        total_wind = report_data['wind_generation'].sum()
                        total_natural_gas = report_data['natural_gas_generation'].sum()
                        total_other = report_data['other_generation'].sum()
                        total_generation = total_solar + total_wind + total_natural_gas + total_other
                        
                        # Generation mix pie chart
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
                        
                        # Generation by source over time
                        st.markdown("### Generation by Source Over Time")
                        
                        fig = px.area(
                            report_data,
                            x='timestamp',
                            y=['solar_generation', 'wind_generation', 'natural_gas_generation', 'other_generation'],
                            title='Generation Mix Over Time',
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
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Weather Impact
                    if include_weather_impact:
                        st.markdown("## Weather Impact Analysis")
                        
                        # Temperature vs Load scatter plot
                        fig = px.scatter(
                            report_data,
                            x='temperature',
                            y='load_mw',
                            color='timestamp',
                            title='Temperature vs Load',
                            labels={
                                'temperature': 'Temperature (°F)',
                                'load_mw': 'Load (MW)',
                                'timestamp': 'Hour of Day'
                            },
                            trendline='ols'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Temperature vs Price scatter plot
                        fig = px.scatter(
                            report_data,
                            x='temperature',
                            y='lmp_price',
                            color='timestamp',
                            title='Temperature vs Price',
                            labels={
                                'temperature': 'Temperature (°F)',
                                'lmp_price': 'Price ($/MWh)',
                                'timestamp': 'Hour of Day'
                            },
                            trendline='ols'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Temperature vs Price scatter plot
                        fig = px.scatter(
                            report_data,
                            x='temperature',
                            y='lmp_price',
                            color='timestamp',
                            title='Temperature vs Price',
                            labels={
                                'temperature': 'Temperature (°F)',
                                'lmp_price': 'Price ($/MWh)',
                                'timestamp': 'Hour of Day'
                            },
                            trendline='ols'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Temperature trend
                        st.markdown("### Temperature Trend")
                        
                        fig = px.line(
                            report_data,
                            x='timestamp',
                            y='temperature',
                            title='Temperature Trend',
                            labels={'temperature': 'Temperature (°F)', 'timestamp': 'Time'}
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=report_data['timestamp'],
                            y=report_data['temperature'].rolling(window=24).mean(),
                            mode='lines',
                            name='24-hour Moving Average',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecasts
                    if include_forecast_section and include_forecast:
                        st.markdown("## Forecasts")
                        
                        # Get forecast data
                        forecast_end_date = datetime.now() + timedelta(days=forecast_days)
                        forecast_data = st.session_state.forecast_data[
                            st.session_state.forecast_data['timestamp'] <= forecast_end_date
                        ]
                        
                        # Load forecast
                        st.markdown("### Load Forecast")
                        
                        fig = px.line(
                            forecast_data,
                            x='timestamp',
                            y='load_forecast',
                            title='Load Forecast',
                            labels={'load_forecast': 'Load (MW)', 'timestamp': 'Time'}
                        )
                        
                        # Add confidence intervals
                        upper_bound = forecast_data['load_forecast'] * (1 + 0.05)
                        lower_bound = forecast_data['load_forecast'] * (1 - 0.05)
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_data['timestamp'],
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,0,255,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_data['timestamp'],
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,0,255,0)',
                            fillcolor='rgba(0,0,255,0.1)',
                            name='95% Confidence Interval'
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Price forecast
                        st.markdown("### Price Forecast")
                        
                        fig = px.line(
                            forecast_data,
                            x='timestamp',
                            y='price_forecast',
                            title='Price Forecast',
                            labels={'price_forecast': 'Price ($/MWh)', 'timestamp': 'Time'}
                        )
                        
                        # Add confidence intervals
                        upper_bound = forecast_data['price_forecast'] * (1 + 0.1)
                        lower_bound = forecast_data['price_forecast'] * (1 - 0.1)
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_data['timestamp'],
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line_color='rgba(255,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_data['timestamp'],
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(255,0,0,0)',
                            fillcolor='rgba(255,0,0,0.1)',
                            name='95% Confidence Interval'
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    if include_recommendations:
                        st.markdown("## Recommendations")
                        
                        # Generate synthetic recommendations based on data
                        avg_price = report_data['lmp_price'].mean()
                        max_price = report_data['lmp_price'].max()
                        avg_load = report_data['load_mw'].mean()
                        peak_load = report_data['load_mw'].max()
                        
                        # Price-based recommendations
                        st.markdown("### Market Strategy Recommendations")
                        
                        if avg_price > 45:
                            st.info("• Consider load reduction during peak price hours to minimize costs")
                            st.info("• Evaluate hedging strategies to mitigate exposure to high prices")
                        else:
                            st.info("• Current prices are favorable - consider locking in longer-term contracts")
                            st.info("• Monitor for potential price increases as seasonal patterns shift")
                        
                        if max_price > 80:
                            st.warning("• Implement automated demand response for extreme price events")
                            st.warning("• Review peak pricing events to identify patterns and prepare mitigation strategies")
                        
                        # Load-based recommendations
                        st.markdown("### Load Management Recommendations")
                        
                        load_factor = avg_load / peak_load * 100 if peak_load > 0 else 0
                        
                        if load_factor < 70:
                            st.info("• Poor load factor indicates opportunity for load shifting to reduce peaks")
                            st.info("• Consider energy storage solutions to improve load factor")
                        else:
                            st.info("• Good load factor - continue current load management practices")
                        
                        # Generation mix recommendations
                        st.markdown("### Generation Mix Recommendations")
                        
                        renewable_pct = (report_data['solar_generation'].sum() + report_data['wind_generation'].sum()) / \
                                        report_data['load_mw'].sum() * 100 if report_data['load_mw'].sum() > 0 else 0
                        
                        if renewable_pct < 15:
                            st.info("• Consider increasing renewable energy procurement to meet sustainability goals")
                            st.info("• Evaluate on-site solar generation opportunities")
                        else:
                            st.info("• Strong renewable mix - continue to monitor integration and reliability")
                    
                    # Raw Data Tables
                    if include_raw_data:
                        st.markdown("## Raw Data")
                        
                        # Daily summary table
                        st.markdown("### Daily Summary")
                        
                        daily_data = report_data.set_index('timestamp').resample('D').agg({
                            'load_mw': ['mean', 'max', 'min'],
                            'lmp_price': ['mean', 'max', 'min'],
                            'temperature': ['mean', 'max', 'min'],
                            'solar_generation': 'sum',
                            'wind_generation': 'sum',
                            'natural_gas_generation': 'sum',
                            'other_generation': 'sum'
                        }).reset_index()
                        
                        # Flatten multi-level column names
                        daily_data.columns = [' '.join(col).strip() for col in daily_data.columns.values]
                        
                        # Rename columns for clarity
                        daily_data = daily_data.rename(columns={
                            'timestamp ': 'Date',
                            'load_mw mean': 'Avg Load (MW)',
                            'load_mw max': 'Max Load (MW)',
                            'load_mw min': 'Min Load (MW)',
                            'lmp_price mean': 'Avg Price ($/MWh)',
                            'lmp_price max': 'Max Price ($/MWh)',
                            'lmp_price min': 'Min Price ($/MWh)',
                            'temperature mean': 'Avg Temp (°F)',
                            'temperature max': 'Max Temp (°F)',
                            'temperature min': 'Min Temp (°F)',
                            'solar_generation sum': 'Solar (MWh)',
                            'wind_generation sum': 'Wind (MWh)',
                            'natural_gas_generation sum': 'Natural Gas (MWh)',
                            'other_generation sum': 'Other (MWh)'
                        })
                        
                        # Format date column
                        daily_data['Date'] = daily_data['Date'].dt.strftime('%Y-%m-%d')
                        
                        # Round numeric columns
                        for col in daily_data.columns:
                            if col != 'Date':
                                daily_data[col] = daily_data[col].round(2)
                        
                        st.dataframe(daily_data, use_container_width=True)
                        
                        # Hourly data table (limited rows for readability)
                        st.markdown("### Hourly Data Sample")
                        
                        hourly_sample = report_data.copy()
                        hourly_sample['timestamp'] = hourly_sample['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Select relevant columns and limit rows
                        hourly_display = hourly_sample[
                            ['timestamp', 'load_mw', 'lmp_price', 'temperature', 
                             'solar_generation', 'wind_generation', 'congestion_index']
                        ].head(24)  # Show first 24 hours
                        
                        # Rename columns for clarity
                        hourly_display = hourly_display.rename(columns={
                            'timestamp': 'Time',
                            'load_mw': 'Load (MW)',
                            'lmp_price': 'Price ($/MWh)',
                            'temperature': 'Temp (°F)',
                            'solar_generation': 'Solar (MW)',
                            'wind_generation': 'Wind (MW)',
                            'congestion_index': 'Congestion'
                        })
                        
                        st.dataframe(hourly_display, use_container_width=True)
                    
                    # Report footer
                    st.markdown("---")
                    st.markdown(f"*Report generated by Charlottesville Power Analyst AI Assistant on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
                
                # Provide download options
                st.markdown("### Download Options")
                
                # Create Excel report
                buffer = io.BytesIO()
                
                # Create a Pandas Excel writer using the BytesIO object
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Write each dataframe to a different worksheet
                    report_data.to_excel(writer, sheet_name='Hourly Data', index=False)
                    
                    if include_raw_data:
                        daily_data.to_excel(writer, sheet_name='Daily Summary', index=False)
                    
                    if include_forecast_section and include_forecast:
                        forecast_data.to_excel(writer, sheet_name='Forecast Data', index=False)
                
                # Get the Excel data
                excel_data = buffer.getvalue()
                
                # Create download link for Excel
                b64_excel = base64.b64encode(excel_data).decode()
                excel_filename = f"charlottesville_energy_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
                excel_href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{excel_filename}">Download Excel Report</a>'
                
                # Create download link for CSV
                csv_data = report_data.to_csv(index=False)
                b64_csv = base64.b64encode(csv_data.encode()).decode()
                csv_filename = f"charlottesville_energy_data_{datetime.now().strftime('%Y%m%d')}.csv"
                csv_href = f'<a href="data:file/csv;base64,{b64_csv}" download="{csv_filename}">Download CSV Data</a>'
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(excel_href, unsafe_allow_html=True)
                with col2:
                    st.markdown(csv_href, unsafe_allow_html=True)
    