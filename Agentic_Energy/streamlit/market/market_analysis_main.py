import streamlit as st
from market.market_analysis import market_analysis_main

def main():
    st.title("Energy Market Analysis Dashboard")
    
    # Run the market analysis
    market_analysis_main()

if __name__ == "__main__":
    main()