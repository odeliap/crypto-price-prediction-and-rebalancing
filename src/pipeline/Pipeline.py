"""
End-to-end pipeline for price prediction and index fund rebalancing.
"""

# ----------- Libraries -----------
import logging

import streamlit as st

import yfinance as yf

from Utils import *
from src.scraper.NewsApiScraper import main as news_scraper
from src.sentiment_analysis.SentimentAnalyzer import SentimentAnalyzer
from src.models.SentimentPriceLSTMModel import predict
"""
from src.index_fund_rebalancing.AlgorithmicTrading import main as rebalancingAlgorithm
from src.index_fund_rebalancing.Evaluation import report_evaluation_metrics
"""

# Set logging level
logging.basicConfig(level=logging.INFO)

# ----------- Constants -----------

available_cryptocurrencies = {'Bitcoin': 'BTC-USD'}
# available_cryptocurrencies = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Solana': 'SOL-USD'}

crypto_names = available_cryptocurrencies.keys()

n_steps_in = 30
n_steps_out = 15

numStocks = 3
numRev = 1

days_to_subtract = 30

# Set start and end dates for scraping previous prices and news
end_date_datetime = datetime.today()
start_date_datetime = end_date_datetime - timedelta(days=days_to_subtract)

# Get string formatted start and end dates
date_format = '%Y-%m-%d'
end_date_str = end_date_datetime.strftime(date_format)
start_date_str = start_date_datetime.strftime(date_format)

# Temporary directories to be created
temp_dir = 'temporary'
price_subdir = f'{temp_dir}/prices'
news_subdir = f'{temp_dir}/news'
clean_subdir = f'{temp_dir}/clean'
sentiment_subdir = f'{temp_dir}/sentiment'
predicted_subdir = f'{temp_dir}/predicted'

temporary_directories = [temp_dir, price_subdir, news_subdir, clean_subdir, sentiment_subdir, price_subdir]

# Paths to model saved files
modelSavedPath = f'src/models/outputs/models/SentimentPriceLSTMModel'
ssScalerSavedPath = f'src/models/outputs/scalers/SentimentPriceLSTMSsScaler'
mmScalerSavedPath = f'src/models/outputs/scalers/SentimentPriceLSTMMmScaler'

# ----------- Main -----------


def pipeline(
        cryptocurrencies: dict,
        start_date: datetime,
        end_date: datetime,
        date_str_format: str
):
    """
    Run pipeline end-to-end.

    Parameters
    __________
    cryptocurrencies : dictionary
        Mapping of cryptocurrency name to slug for available coins.
    start_date : datetime
        Date to start scraping historical data from.
    end_date : datetime
        Date to stop scraping historical data to.
    date_str_format : str
        Date format for formatting datetime objects.
    """
    with st.spinner(f'Retrieving previous daily prices'):
        # For each available coin
        for coin in crypto_names:
            # Use yfinance to download last 30 days of historical prices
            prev_prices = yf.download(cryptocurrencies.get(coin), start=start_date_str, end=end_date_str, interval='1d')
            # Clean the price dataframe
            prev_prices_clean = clean_prices_dataframe(prev_prices, start_date, end_date, date_str_format)
            # Save historical prices dataframe to csv to the prices subdirectory
            prev_prices_clean.to_csv(f'{price_subdir}/{coin.lower()}_prev_prices.csv', index=False)
    with st.spinner(f'Scraping news'):
        # For each available coin
        for coin in crypto_names:
            # Use News API to get last 30 days of related news
            news_dataframe = news_scraper(coin, start_date, end_date)
            # Save related news dataframe to csv to the news subdirectory
            news_dataframe.to_csv(f'{news_subdir}/{coin.lower()}_news.csv', index=False)
    with st.spinner(f'Cleaning input data for sentiment analysis'):
        for coin in crypto_names:
            # Load the price and news dataframes
            price_dataframe = pd.read_csv(f'{price_subdir}/{coin}_prev_prices.csv')
            news_dataframe = pd.read_csv(f'{news_subdir}/{coin}_news.csv')
            # Combine the price and news dataframes
            combined_dataframe = combine_dataframes(price_dataframe, news_dataframe)
            # Save clean combined dataframe to the clean subdirectory
            combined_dataframe.to_csv(f'{clean_subdir}/{coin}_combined.csv', index=False)
    with st.spinner(f'Getting sentiment for found news'):
        for coin in crypto_names:
            # Read input dataframe from clean subdirectory
            input_dataframe = pd.read_csv(f'{clean_subdir}/{coin}_combined.csv')
            # Perform sentiment analysis
            sentiment_dataframe = SentimentAnalyzer(input_dataframe).dataframe
            # Save sentiment dataframe to the sentiment subdirectory
            sentiment_dataframe.to_csv(f'{sentiment_subdir}/{coin}_sentiment.csv', index=False)
    with st.spinner(f'Retrieving predicted prices'):
        for coin in crypto_names:
            # Load the sentiment dataframe
            sentiment_data, open_prices = \
                format_sentiment_input_for_predictions(f'{sentiment_subdir}/{coin}_sentiment.csv')
            # Make predictions
            predictions = predict(
                sentiment_data,
                open_prices,
                coin,
                n_steps_in,
                n_steps_out,
                modelSavedPath,
                ssScalerSavedPath,
                mmScalerSavedPath
            )
            # Save predictions to the predictions subdirectory
            predictions.to_csv(f'{predicted_subdir}/{coin}_predictions.csv')


"""
    # TODO: Combine predicted prices into single csv file and save to predicted prices subdirectory
    predicted_prices = pd.DataFrame()

    rebalanced_portfolio, stock_returns = rebalancingAlgorithm(predicted_prices, numStocks, numRev)
    report_evaluation_metrics(rebalanced_portfolio, start_date, end_date)
"""


def main():
    """
    Run pipeline.
    """
    make_dirs(temporary_directories)
    pipeline(available_cryptocurrencies, start_date_datetime, end_date_datetime, date_format)
    # delete_dirs(temporary_directories)
