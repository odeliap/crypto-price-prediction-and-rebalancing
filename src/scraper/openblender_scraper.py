"""
Data scraper for news sources and crypto prices using Open Blender API
"""

# ------------- Libraries -------------

import OpenBlender
from io import StringIO
import pandas as pd
import json

import logging

from util import dataframe_to_csv

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

ACTION = 'API_getObservationsFromDataset'

PRICE_FILEPATH = 'datasets/price/openblender_price.csv'
NEWS_FILEPATH = 'datasets/news/openblender_news.csv'


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
        :return:
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


if __name__ == "__main__":
    scraper = OpenBlenderScraper()