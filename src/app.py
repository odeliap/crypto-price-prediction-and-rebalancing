"""
Simple UI made with streamlit for easy visualization of the proposed workflow.
"""

# ----------- Libraries -----------
import logging

import streamlit as st
import pandas as pd

import time
from datetime import datetime, timedelta

import yfinance as yf

from scraper.NewsApiScraper import main as newsScraper
from sentiment_analysis.SentimentAnalyzer import SentimentAnalyzer
from models.SentimentPriceLSTMModel import predict
from index_fund_rebalancing.AlgorithmicTrading import main as rebalancingAlgorithm
from index_fund_rebalancing.AlgorithmicTrading import CAGR, sharpe_ratio, maximum_drawdown

# Set logging level
logging.basicConfig(level=logging.INFO)

# ----------- Constants -----------

st.header('Crytpo Price Prediction and Index Fund Rebalancing')

cryptocurrencies = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Solana': 'SOL-USD'}

days_to_subtract = 100
date_format = '%Y-%m-%d'

n_steps_in = 30

numStocks = 3
numRev = 1

# ----------- App -----------

predicted_prices = pd.DataFrame()

# Use yfinance to get dataframe of previous prices for these coins
end_date = datetime.today()
start_date = end_date - timedelta(days=days_to_subtract)

end_date_str = end_date.strftime(date_format)
start_date_str = start_date.strftime(date_format)

for coin in cryptocurrencies.keys():
    with st.spinner(f'Retrieving previous daily prices for {coin}'):
        prev_stock_prices = yf.download(cryptocurrencies.get(coin), start=start_date_str, end=end_date_str, interval='1d')
        prev_stock_prices_clean = prev_stock_prices.dropna()
    with st.spinner(f'Scraping {coin} news'):
        dataframe = newsScraper(coin)
        time.sleep(10)
    with st.spinner(f'Getting sentiment for found news'):
        # TODO: format previous stock prices with sentiment dataframe to make input to sentiment analyzer
        sentiment_dataframe = SentimentAnalyzer(dataframe).dataframe
        time.sleep(10)
    with st.spinner(f'Retrieving predicted price for {coin}'):
        predictions = predict(sentiment_dataframe, coin, n_steps_in)
        time.sleep(10)


rebalanced_portfolio, stock_returns = rebalancingAlgorithm(predicted_prices, numStocks, numRev)

logging.info("Rebalanced Portfolio Performance")
logging.info("CAGR: " + str(CAGR(rebalanced_portfolio, start_date, end_date)))
logging.info("Sharpe Ratio: " + str(sharpe_ratio(rebalanced_portfolio, 0.03, start_date, end_date)))
logging.info("Maximum Drawdown: " + str(maximum_drawdown(rebalanced_portfolio)))