"""
Data scraper for news sources using Python Crypto News API
"""

# ------------- Libraries -------------
import logging

import pandas as pd

from datetime import datetime

from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

from Utils import dataframe_to_csv

from crypto_news_api import CryptoControlAPI

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

news_api_key = 'gvpifrrlnsffapoedsbpc0e7vucofq9saqnqjnsm'
market_api_key = '1b7fcdab-bbfd-4753-840c-e8c853121fd7'

column_names = ['title', 'text', 'timestamp', 'coin']
coin_id_map_col_names = ['slug', 'name', 'symbol']

headlines_filepath = f'datasets/news/processed/cryptonewsapi_news.csv'
coin_id_map_filepath = 'datasets/util/cryptonewsapi_id_map.csv'

language = "en"

# ------------- Class -------------

class CryptoNewsApiScraper:
    """
    Class to scrape news using python crypto news API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate a crypto news api scraper object.
        """
        self.newsapi = CryptoControlAPI(apiKey=news_api_key) # Connect to the CrytpoControl API

        self.headlines = pd.DataFrame(columns=column_names)

        self.coin_ids = self.get_all_crypto_coin_ids()


    def get_top_news(self) -> None:
        """
        Calls crypto news api to get top headlines and appends to self.headlines
        """
        data = json.loads(self.newsapi.getTopNews(language))['data']
        title = data['title']
        text = data['text']
        timestamp = datetime.utcfromtimestamp(int(data['date'])).strftime("%Y-%m-%d")
        tickers = ','.join(data['tickers'])
        row = {'title': title, 'text': text, 'timestamp': timestamp, 'coin': tickers}
        self.headlines = self.headlines.append(row)


    def get_top_news_by_coin(self) -> None:
        """
        Calls crypto news api to get top headlines by cryptocurrency and append to self.headlines
        """
        slugs = self.coin_ids['slug'].tolist()
        for slug in slugs:
            data = json.loads(self.newsapi.getTopNewsByCoin(slug))['data']
            title = data['title']
            text = data['text']
            timestamp = datetime.utcfromtimestamp(int(data['date'])).strftime("%Y-%m-%d")
            row = {'title': title, 'text': text, 'timestamp': timestamp, 'coin': slug}
            self.headlines = self.headlines.append(row)


    @staticmethod
    def get_all_crypto_coin_ids() -> pd.DataFrame:
        """
        Calls Coinmarketcapi API to get mapping of coin name to id

        Returns
        _______
        coin_ids: dictionary
            Mapping between cryptocurrency name to id.
        """
        coin_ids = pd.DataFrame(columns=coin_id_map_col_names)

        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/map' # Coinmarketcap API url for getting cryptocurrency mapping of name to id

        parameters = { 'limit': '50', 'sort': 'cmc_rank' }

        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': market_api_key
        }

        session = Session()
        session.headers.update(headers)

        try:
            response = session.get(url, params=parameters)

            info = json.loads(response.text)['data']

            for coin in info:
                slug = coin['slug']
                name = coin['name']
                symbol = coin['symbol']
                row = {'slug': slug, 'name': name, 'symbol': symbol}
                logging.info(row)
                coin_ids = coin_ids.append(row, ignore_index=True)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            logging.error(e)

        return coin_ids


if __name__ == "__main__":
    """
    Instantiate a new scraper object and scrape for all the available coins
    """
    scraper = CryptoNewsApiScraper()
    scraper.get_top_news()
    scraper.get_top_news_by_coin()
    dataframe_to_csv(scraper.headlines, headlines_filepath)
    dataframe_to_csv(scraper.coin_ids, coin_id_map_filepath)