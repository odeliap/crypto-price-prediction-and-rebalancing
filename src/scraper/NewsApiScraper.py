"""
Data scraper for news sources using News API
"""

# ------------- Libraries -------------
import logging
import math

import pandas as pd

from datetime import datetime

from newsapi import NewsApiClient

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

api_key = '681ea4f3f4de4f29a08687c9f692beed'

categories = ['business', 'technology'] # categories to query

country = 'us'
date_format = '%Y-%m-%d'
#end_date = datetime.today().strftime(date_format) # Set scraping end date
#start_date = datetime.strptime(end_date, date_format) - relativedelta(days=30) # Set scraping start date
sort_by = 'popularity'
page_size = 100

column_names = ['text', 'timestamp']

# ------------- Class -------------

class NewsApiScraper:
    """
    Class to scrape news using news API
    """

    def __init__(
        self,
        coin: str,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        Instantiate a news api scraper object.

        Parameters
        __________
        coin : string
            Coin to search for news on.
        start_date : datetime
            Date to begin scraping news from.
        end_date : datetime
            Date to stop scraping news on.
        """
        self.newsapi = NewsApiClient(api_key=api_key)

        self.headlines = pd.DataFrame(columns=column_names)
        self.daily_api_requests = 100

        self.keywords = coin
        self.sources = self.get_all_sources()

        self.start_date = start_date
        self.end_date = end_date


    def get_all_headlines(self) -> None:
        """
        Gets all the headlines across categories of interest.
        Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_all_headlines()
        # Below commented out code requires purchasing premium plan
        # to increase querying limit
        """if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                remaining_pages = self.remaining_pages(total_results)
                for page in range(2, remaining_pages + 2):
                    if self.daily_api_requests > 0:
                        self.base_get_all_headlines(page)
                        self.daily_api_requests -= 1"""


    def get_top_headlines(self) -> None:
        """
        Gets all top headlines across categories of interest.
        Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_top_headlines()
        # Below commented out code requires purchasing premium plan
        # to increase querying limit
        """if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                remaining_pages = self.remaining_pages(total_results)
                for page in range(2, remaining_pages + 2):
                    if self.daily_api_requests > 0:
                        self.base_get_top_headlines(page)
                        self.daily_api_requests -= 1"""


    def base_get_all_headlines(self, page: int = 1) -> dict:
        """
        Send news api get everything request.

        Parameters
        __________
        page : int, default 1
            Current page.

        Returns
        _______
        headlines : dictionary
            News api result.
        """
        headlines = self.newsapi.get_everything(
            q=self.keywords,
            sources=self.sources,
            from_param=self.start_date,
            to=self.end_date,
            sort_by=sort_by,
            page_size=page_size,
            page=page
        )
        self.daily_api_requests = self.daily_api_requests - 1
        self.news_api_dict_to_dataframe(headlines)

        return headlines


    def base_get_top_headlines(self, page: int = 1) -> dict:
        """
        Send news api top headlines request.

        Parameters
        __________
        page : int, default 1
            Current page.

        Results
        _______
        headlines : dictionary
            News api result.
        """
        headlines = self.newsapi.get_top_headlines(
                q=self.keywords,
                sources=self.sources,
                page_size=page_size,
                page=page
            )
        self.daily_api_requests = self.daily_api_requests - 1
        self.news_api_dict_to_dataframe(headlines)

        return headlines


    def get_all_sources(self) -> str:
        """
        Gets all the sources available across the categories of interest.

        Returns
        _______
        sources : string
            All the sources available across the categories of interest.
        """
        combined_sources = []

        for category in categories:
            sources_info = self.newsapi.get_sources(
                category = category,
                country = country
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
        Sorts a news api dictionary object to create a pandas dataframe.
        Adds this dictionary content to self.headlines.

        Parameters
        __________
        dictionary : dictionary
            News api result.
        """
        logging.info(f'dictionary: {dictionary}')
        articles = dictionary.get('articles')
        for article in articles:
            title = article.get('title')
            description = article.get('description')
            publishedAt = article.get('publishedAt').replace('T', ' ').replace('Z', '')
            space = publishedAt.index(' ')
            timestamp = publishedAt[0:space]
            row = {'text': title + ' ' + description, 'timestamp': timestamp}
            logging.info(f'row: {row}')
            self.headlines = self.headlines.append(row, ignore_index=True)
        logging.info(f'updated headlines: {self.headlines}')


    @staticmethod
    def remaining_pages(total_results: int) -> int:
        """
        Get remaining pages to sort through.

        Parameters
        __________
        total_results : int
            Total number of results returned from news api query.

        Returns
        _______
        remaining_pages : int
            Remaining number of pages in the query.
        """
        return math.ceil((total_results - 100)/100)


def main(coin: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Retrieve and return news headlines for coin of interest.

    Parameters
    __________
    coin : string
        Coin to search for news on.
    start_date : datetime
        Date to begin scraping news from.
    end_date : datetime
        Date to stop scraping news on.

    Returns
    _______
    headlines : pandas dataframe
        News headlines for the given coin.
    """
    scraper = NewsApiScraper(coin, start_date, end_date)
    scraper.get_all_headlines()
    scraper.get_top_headlines()
    headlines = scraper.headlines
    return headlines