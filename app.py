# ----------- Libraries -----------

import streamlit as st

import time

# ----------- App -----------

st.header('Crytpo Price Prediction and Index Fund Rebalancing')

cryptocurrencies = ['Bitcoin', 'Ethereum', 'Solana']

user_coins = st.multiselect(cryptocurrencies)

predicted_prices = dict()
predicted_price_delta = dict()
holding_amounts = dict()

for coin in user_coins:
    holding = st.number_input(f'How much {coin} do you own?')
    holding_amounts[coin] = holding
    # TODO: get previous price for coin

    with st.spinner(f'Scraping {coin} news'):
        # TODO: 1) scrape related data
        time.sleep(10)
    with st.spinner(f'Retrieving predicted price for {coin}'):
        # TODO: 2) input date and scraped data into model to retrieve predicted price
        # TODO: 3) save predicted price and price delta to respective map
        time.sleep(10)

# TODO: input predicted_prices, predicted_price_delta, and holding_amounts to rebalancing algorithm