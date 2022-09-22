"""
Data scraper for news sources using News API
"""

# ------------- Libraries -------------
import logging
import math

import pandas as pd

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from util import dataframe_to_csv
from newsapi import NewsApiClient

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

API_KEY = '681ea4f3f4de4f29a08687c9f692beed'

CATEGORIES = ['business', 'technology']

# list of coins of interest from: https://coinmarketcap.com
COINS = ['Tether', 'USD Coin', 'BNB', 'Binance USD', 'Cardano', 'XRP', 'Solana', 'Dogecoin',
            'Polkadot', 'Polygon', 'Shiba Inu', 'Dai', 'TRON', 'Avalanche', 'UNUS SED LEO',
            'Wrapped Bitcoin', 'Uniswap', 'Etherum Classic', 'Litecoin', 'Cosmos', 'Chainlink',
            'FTX Token', 'NEAR Protocol', 'Cronos', 'Monero', 'Stellar', 'Bitcoin Cash',
            'Algorand', 'Flow', 'VeChain', 'Filecoin', 'Internet Computer', 'Decentraland', 'EOS',
            'ApeCoin', 'The Sandbox', 'Tezos', 'Hedera', 'Chiliz', 'Aave', 'Axie Infinity',
            'Elrond', 'Theta Network', 'Quant', 'TrueUSD', 'Bitcoin SV', 'Zcash', 'Pax Dollar']

# list of keywords built off: https://time.com/nextadvisor/investing/cryptocurrency/crypto-terms-you-should-know-before-investing/
KEYWORDS = ['Altcoin', 'Bitcoin', 'Block', 'Blockchain', 'Coin', 'Coinbase', 'Cold Wallet',
            'Cold Storage', 'Crypto', 'Cryptocurrency', 'Decentralization', 'DeFi', 'DApps',
            'Digital Gold', 'Ethereum', 'Exchange', 'Fork', 'Genesis Block', 'HODL', 'Halving',
            'Hash', 'Hot Wallet', 'Initial Coin Offering', 'ICO', 'Market Capitalization',
            'Mining', 'Node', 'Non-fungible Tokens', 'NFTs', 'Peer-to-peer', 'Public Key',
            'Private Key', 'Satoshi Nakomoto', 'Smart Contract', 'Stablecoin', 'Digital Fiat',
            'Token', 'Vitalik Buterin', 'Wallet']

COUNTRY = 'us'
DATE_FORMAT = '%Y-%m-%d'
END_DATE = datetime.today().strftime(DATE_FORMAT) # Set scraping end date
START_DATE = datetime.strptime(END_DATE, DATE_FORMAT) - relativedelta(months=1) # Set scraping start date
SORT_BY = 'popularity'
PAGE_SIZE = 100

COLUMN_NAMES = ['title', 'text', 'timestamp', 'coin']

today = date.today()
TODAY = today.strftime("%m-%d-%Y")

FILEPATH = f'datasets/news/processed/newsapi_news.csv'

# ------------- Class -------------

class NewsApiScraper:
    """
    Class to scrape news using news API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate a news api scraper object.
        """
        self.newsapi = NewsApiClient(api_key=API_KEY)

        self.headlines = pd.DataFrame(columns=COLUMN_NAMES)
        self.daily_api_requests = 100

        self.keywords = 'crypto' # TODO: find way to query on all keywords and coins
        # self.keywords = ','.join(KEYWORDS + COINS)
        self.sources = self.get_all_sources()


    def get_all_headlines(self) -> None:
        """
        Gets all the headlines across categories of interest. Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_all_headlines()
        # TODO: find way to access more results (increase plan?)
        """
        if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                if self.daily_api_requests > 0:
                    remaining_pages = self.remaining_pages(total_results)
                    for page in range(2, remaining_pages + 2):
                        self.base_get_all_headlines(page)
        """


    def get_top_headlines(self) -> None:
        """
        Gets all top headlines across categories of interest. Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_top_headlines()
        # TODO: find way to access more results (increase plan?)
        """
        if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                while self.daily_api_requests > 0:
                    remaining_pages = self.remaining_pages(total_results)
                    for page in range(2, remaining_pages + 2):
                        self.base_get_top_headlines(page)
        """


    def base_get_all_headlines(self, page: int = 1) -> dict:
        """
        Send news api get everything request.

        :param page: defaulted to 1
        :type int

        :return headlines: news api dictionary result
        :rtype dict
        """
        headlines = self.newsapi.get_everything(
            q=self.keywords,
            sources=self.sources,
            from_param=START_DATE,
            to=END_DATE,
            sort_by=SORT_BY,
            page_size=PAGE_SIZE,
            page=page
        )
        self.daily_api_requests = self.daily_api_requests - 1
        self.news_api_dict_to_dataframe(headlines)

        return headlines


    def base_get_top_headlines(self, page: int = 1) -> dict:
        """
        Send news api top headlines request.

        :param page: defaulted to 1
        :type int

        :return headlines: news api dictionary result
        :rtype dict
        """
        headlines = self.newsapi.get_top_headlines(
                q=self.keywords,
                sources=self.sources,
                page_size=PAGE_SIZE,
                page=page
            )
        self.daily_api_requests = self.daily_api_requests - 1
        self.news_api_dict_to_dataframe(headlines)

        return headlines


    def get_all_sources(self) -> str:
        """
        Gets all the sources available across the categories of interest.

        :return sources: all the sources
        :rtype str
        """
        combined_sources = []

        for category in CATEGORIES:
            sources_info = self.newsapi.get_sources(
                category = category,
                country = COUNTRY
            )
            if sources_info is not None:
                sources = sources_info.get('sources')
                for source in sources:
                    id = source.get('id')
                    combined_sources.append(id)

        logging.info(f'Found sources: {combined_sources}')

        sources = ','.join(combined_sources)
        return sources


    def news_api_dict_to_dataframe(self, dictionary) -> None:
        """
        Sorts a news api dictionary object to create a pandas dataframe. Adds this dictionary content to
        self.headlines.

        :param dictionary: news api Python dictionary result.
        :type dict
        """
        logging.info(f'dictionary: {dictionary}')
        articles = dictionary.get('articles')
        for article in articles:
            title = article.get('title')
            description = article.get('description')
            publishedAt = article.get('publishedAt').replace('T', ' ').replace('Z', '')
            space = publishedAt.index(' ')
            timestamp = publishedAt[0:space]
            row = {'title': title, 'text': description, 'timestamp': timestamp, 'coin': ''}
            logging.info(f'row: {row}')
            self.headlines = self.headlines.append(row, ignore_index=True)
        logging.info(f'updated headlines: {self.headlines}')


    @staticmethod
    def remaining_pages(total_results: int) -> int:
        """
        Get remaining pages to sort through.

        :param total_results: total results returned from news api query
        :type int

        :return remaining_pages: remaining pages in query
        :type int
        """
        return math.ceil((total_results - 100)/100)


if __name__ == "__main__":
    scraper = NewsApiScraper()
    scraper.get_all_headlines()
    scraper.get_top_headlines()
    dataframe_to_csv(scraper.headlines, FILEPATH)