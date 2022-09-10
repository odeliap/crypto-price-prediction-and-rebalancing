"""
Data scraper for news sources and crypto price using Open Blender API
"""

# ------------- Libraries -------------

import OpenBlender
from io import StringIO
import pandas as pd
import json

import logging

import datetime

from util import dataframe_to_csv

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

ACTION = 'API_getObservationsFromDataset'

PRICE_FILEPATH = 'datasets/price/unprocessed/openblender_crypto_index_prices.csv'
NEWS_FILEPATH = 'datasets/news/unprocessed/openblender_news.csv'

CLEAN_PRICE_FILEPATH = 'datasets/price/processed/openblender_crypto_index_prices.csv'
CLEAN_NEWS_FILEPATH = 'datasets/news/processed/openblender_news.csv'

# ------------- Class -------------

class OpenBlenderScraper:
    """
    Class to scrape news using Open Blender API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate a news scraper object.
        """
        self.save_price_dataset()
        self.save_news_tweet_dataset()
        self.clean_and_save_price_dataset()
        self.clean_and_save_tweet_dataset()


    def save_price_dataset(self) -> None:
        """
        Class to save price dataset 'Cryptoindex.com 100 Price' from Open Blender
        """
        parameters = {
            'token': '63163a459516292b27215ea3IU1IFqxx7KdHD5fdPdfmgA1FijH8zF',
            'id_user': '63163a459516292b27215ea3',
            'id_dataset': '5db8ff979516291755e7d09b'
        }

        dataframe = pd.read_json(StringIO(json.dumps(OpenBlender.call(ACTION, parameters)['sample'])),
                                       convert_dates=False,
                                       convert_axes=False).sort_values('timestamp', ascending=False)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.head()
        logging.info(f"Saving 'Cryptoindex.com 100 Price' dataset to {PRICE_FILEPATH}")
        dataframe_to_csv(dataframe, PRICE_FILEPATH)


    def save_news_tweet_dataset(self) -> None:
        """
        Class to save crypto news tweet dataset 'CryptoCurrency News Tweet' from Open Blender
        """
        parameters = {
            'token': '63163a459516292b27215ea3IU1IFqxx7KdHD5fdPdfmgA1FijH8zF',
            'id_user': '63163a459516292b27215ea3',
            'id_dataset': '5ea209c495162936348f13eb'
        }

        dataframe = pd.read_json(StringIO(json.dumps(OpenBlender.call(ACTION, parameters)['sample'])),
                                 convert_dates=False,
                                 convert_axes=False).sort_values('timestamp', ascending=False)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.head()
        logging.info(f"Saving 'CryptoCurrency News Tweet' dataset to {NEWS_FILEPATH}")
        dataframe_to_csv(dataframe, NEWS_FILEPATH)


    def clean_and_save_tweet_dataset(self) -> None:
        """
        Class to save cleaned version of tweet dataset
        """
        save_columns = ['text', 'timestamp']
        news_dataframe = pd.read_csv(NEWS_FILEPATH)
        clean_news_dataframe = news_dataframe[save_columns]
        timestamps = clean_news_dataframe['timestamp']
        clean_timestamps = []
        for timestamp in timestamps:
            clean_time = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
            clean_timestamps.append(clean_time)
        clean_news_dataframe['timestamp'] = clean_timestamps
        clean_news_dataframe['coin'] = ''
        dataframe_to_csv(clean_news_dataframe, CLEAN_NEWS_FILEPATH)


    def clean_and_save_price_dataset(self) -> None:
        """
        Class to save cleaned version of price dataset
        """
        save_columns = ['volume', 'timestamp', 'price', 'high', 'low', 'open', 'change']
        price_dataframe = pd.read_csv(PRICE_FILEPATH)
        clean_price_dataframe = price_dataframe[save_columns]
        timestamps = clean_price_dataframe['timestamp']
        clean_timestamps = []
        for timestamp in timestamps:
            clean_time = datetime.datetime.utcfromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
            clean_timestamps.append(clean_time)
        clean_price_dataframe['timestamp'] = clean_timestamps
        clean_price_dataframe['coin'] = ''
        dataframe_to_csv(clean_price_dataframe, CLEAN_PRICE_FILEPATH)


if __name__ == "__main__":
    scraper = OpenBlenderScraper()