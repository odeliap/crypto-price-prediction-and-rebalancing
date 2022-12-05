"""
Simple UI made with streamlit for easy visualization of the proposed workflow.
"""

# ----------- Libraries -----------
import streamlit as st

from Pipeline import main as run_pipeline

# ----------- App -----------

st.header('Crytpo Price Prediction and Index Fund Rebalancing')
st.write('Cryptocurrencies currently included in fund: Bitcoin, Ethereum, and Solana.')

# Add button for kicking off pipeline
retrieve_predictions = st.button('Kick off price prediction and rebalancing pipeline')

# If the button is clicked run the pipeline
if retrieve_predictions:
    run_pipeline()