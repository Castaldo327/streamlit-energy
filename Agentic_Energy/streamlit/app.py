# app.py
import streamlit as st
import io
import base64
from dashboard.dashboard_daily_briefing import dashboard_daily_briefing
from market.market_analysis_main import main
from forecast.forecast_main import forecasting_and_modeling
from report_generation.report_generation import report_generation

st.set_page_config(
    page_title="Charlottesville Power Analyst Assistant",
    initial_sidebar_state="expanded"
)

if 'update_forecasts' not in st.session_state:
    st.session_state.update_forecasts = False
    
# Define application sections
SECTIONS = [
    "Dashboard & Daily Briefing",
    "Market Analysis Workbench",
    "Forecasting & Modeling Suite",
    "Report Generator"
]

# Sidebar for navigation
with st.sidebar:
    st.title("Power Analyst Assistant")
    st.subheader("Charlottesville, Virginia")
    
    # Navigation
    selected_section = st.radio("Navigation", SECTIONS)
    
    # User information
    st.markdown("---")
    st.info("User: Power Analyst")
    st.info("Region: PJM - Dominion Zone")
    
    # Quick actions
    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("Generate Daily Report"):
        st.success("Report generated!")

# Main content area
st.title(selected_section)

# Function to create a download link for dataframes
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to create a download link for figures
def get_figure_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

if selected_section == "Dashboard & Daily Briefing":
    dashboard_daily_briefing()

elif selected_section == "Market Analysis Workbench":
    main()
    
elif selected_section == "Forecasting & Modeling Suite":
    forecasting_and_modeling()
    
elif selected_section == "Report Generator":
    report_generation()
    
    