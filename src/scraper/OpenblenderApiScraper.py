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

from Utils import dataframe_to_csv

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

action = 'API_getObservationsFromDataset'

price_filepath = 'datasets/price/unprocessed/openblender_crypto_index_prices.csv'
news_filepath = 'datasets/news/unprocessed/openblender_news.csv'

clean_price_filepath = 'datasets/price/processed/openblender_crypto_index_prices.csv'
clean_news_filepath = 'datasets/news/processed/openblender_news.csv'

# ------------- Class -------------

class OpenBlenderApiScraper:
    """
    Class to scrape news using Open Blender API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate an open blender api scraper object.
        """
        self.save_price_dataset()
        self.save_news_tweet_dataset()
        self.clean_and_save_price_dataset()
        self.clean_and_save_tweet_dataset()


    def save_price_dataset(self) -> None:
        """
        Save price dataset 'Cryptoindex.com 100 Price' from Open Blender
        """
        parameters = {
            'token': '63163a459516292b27215ea3IU1IFqxx7KdHD5fdPdfmgA1FijH8zF',
            'id_user': '63163a459516292b27215ea3',
            'id_dataset': '5db8ff979516291755e7d09b'
        }

        dataframe = pd.read_json(StringIO(json.dumps(OpenBlender.call(action, parameters)['sample'])),
                                       convert_dates=False,
                                       convert_axes=False).sort_values('timestamp', ascending=False)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.head()
        logging.info(f"Saving 'Cryptoindex.com 100 Price' dataset to {price_filepath}")
        dataframe_to_csv(dataframe, price_filepath)


    def save_news_tweet_dataset(self) -> None:
        """
        Save crypto news tweet dataset 'CryptoCurrency News Tweet' from Open Blender
        """
        parameters = {
            'token': '63163a459516292b27215ea3IU1IFqxx7KdHD5fdPdfmgA1FijH8zF',
            'id_user': '63163a459516292b27215ea3',
            'id_dataset': '5ea209c495162936348f13eb'
        }

        dataframe = pd.read_json(StringIO(json.dumps(OpenBlender.call(action, parameters)['sample'])),
                                 convert_dates=False,
                                 convert_axes=False).sort_values('timestamp', ascending=False)
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.head()
        logging.info(f"Saving 'CryptoCurrency News Tweet' dataset to {news_filepath}")
        dataframe_to_csv(dataframe, news_filepath)


    def clean_and_save_tweet_dataset(self) -> None:
        """
        Save cleaned version of tweet dataset
        """
        save_columns = ['text', 'timestamp']
        news_dataframe = pd.read_csv(news_filepath)
        clean_news_dataframe = news_dataframe[save_columns]
        timestamps = clean_news_dataframe['timestamp']
        clean_timestamps = []
        for timestamp in timestamps:
            clean_time = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
            clean_timestamps.append(clean_time)
        clean_news_dataframe['timestamp'] = clean_timestamps
        clean_news_dataframe['coin'] = ''
        dataframe_to_csv(clean_news_dataframe, clean_news_filepath)


    def clean_and_save_price_dataset(self) -> None:
        """
        Save cleaned version of price dataset
        """
        save_columns = ['volume', 'timestamp', 'price', 'high', 'low', 'open', 'change']
        price_dataframe = pd.read_csv(price_filepath)
        clean_price_dataframe = price_dataframe[save_columns]
        timestamps = clean_price_dataframe['timestamp']
        clean_timestamps = []
        for timestamp in timestamps:
            clean_time = datetime.datetime.utcfromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
            clean_timestamps.append(clean_time)
        clean_price_dataframe['timestamp'] = clean_timestamps
        clean_price_dataframe['coin'] = ''
        dataframe_to_csv(clean_price_dataframe, clean_price_filepath)


if __name__ == "__main__":
    """
    Perform scraping
    """
    scraper = OpenBlenderApiScraper()