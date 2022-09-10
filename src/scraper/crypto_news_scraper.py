"""
Data scraper for news sources using Python Crypto News API
"""

# ------------- Libraries -------------
import logging

import pandas as pd

from datetime import date, datetime

from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

from util import dataframe_to_csv

from crypto_news_api import CryptoControlAPI

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

NEWS_API_KEY = 'gvpifrrlnsffapoedsbpc0e7vucofq9saqnqjnsm'
MARKET_API_KEY = '1b7fcdab-bbfd-4753-840c-e8c853121fd7'

COLUMN_NAMES = ['title', 'text', 'timestamp', 'coin']
COIN_ID_MAP_COLUMN_NAMES = ['slug', 'name', 'symbol']

today = date.today()
TODAY = today.strftime("%m-%d-%Y")

HEADLINES_FILEPATH = f'datasets/news/processed/cryptonewsapi_news.csv'
COIN_ID_MAP_FILEPATH = 'datasets/util/cryptonewsapi_id_map.csv'

LANGUAGE = "en"

# ------------- Class -------------

class CryptoNewsScraper:
    """
    Class to scrape news using python crypto news API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate a news scraper object.
        """
        self.newsapi = CryptoControlAPI(apiKey=NEWS_API_KEY) # Connect to the CrytpoControl API

        self.headlines = pd.DataFrame(columns=COLUMN_NAMES)

        self.coin_ids = self.get_all_crypto_coin_ids()


    def get_top_news(self) -> None:
        """
        Calls crypto news api to get top headlines and appends to self.headlines
        """
        data = json.loads(self.newsapi.getTopNews(LANGUAGE))['data']
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

        :return coin_ids: Python dictionary mapping cryptocurrency name to id
        :rtype dict
        """
        coin_ids = pd.DataFrame(columns=COIN_ID_MAP_COLUMN_NAMES)

        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/map' # Coinmarketcap API url for getting cryptocurrency mapping of name to id

        parameters = { 'limit': '50', 'sort': 'cmc_rank' }

        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': MARKET_API_KEY
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
    # TODO: fix api call, getting bad request
    scraper = CryptoNewsScraper()
    scraper.get_top_news()
    scraper.get_top_news_by_coin()
    dataframe_to_csv(scraper.headlines, HEADLINES_FILEPATH)
    dataframe_to_csv(scraper.coin_ids, COIN_ID_MAP_FILEPATH)