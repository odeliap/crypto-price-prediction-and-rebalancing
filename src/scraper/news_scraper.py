"""
Data scraper for news sources
"""

# ------------- Libraries -------------
import logging
import math

import time

import os

import pandas as pd

from newsapi import NewsApiClient

# Set logging level
logging.basicConfig(level=logging.INFO)


# ------------- Constants -------------

API_KEY = '681ea4f3f4de4f29a08687c9f692beed'

CATEGORIES = ['business', 'entertainment', 'technology']

KEYWORDS = ['bitcoin', 'cryptocurrency'] # TODO: expand keywords

COUNTRY = 'us'
START_DATE = '2022-08-03'
END_DATE = '2022-09-02'
SORT_BY = 'popularity'
PAGE_SIZE = 100

COLUMN_NAMES = ['title', 'description', 'publishedAt']

FILEPATH = 'outputs/news.csv'

# ------------- Class -------------

class NewsScraper:
    """
    Class to scrape news using news API
    """

    def __init__(
        self
    ) -> None:
        """
        Instantiate a news scraper object.
        """
        self.newsapi = NewsApiClient(api_key=API_KEY)

        self.headlines = pd.DataFrame(columns=COLUMN_NAMES)
        self.daily_api_requests = 100

        self.keywords = ','.join(KEYWORDS)
        self.sources = self.get_all_sources()


    def get_all_headlines(self) -> None:
        """
        Gets all the headlines across categories of interest. Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_all_headlines()
        if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                remaining_pages = self.remaining_pages(total_results)
                for page in range(2, remaining_pages + 2):
                    self.base_get_all_headlines(page)


    def get_top_headlines(self) -> None:
        """
        Gets all top headlines across categories of interest. Results are appended to self.headlines pandas dataframe.
        """
        headlines = self.base_get_top_headlines()
        if headlines is not None:
            total_results = headlines.get('totalResults')
            if total_results > 100:
                remaining_pages = self.remaining_pages(total_results)
                for page in range(2, remaining_pages + 2):
                    self.base_get_top_headlines(page)


    def base_get_all_headlines(self, page: int = 1) -> dict:
        """
        Send news api get everything request.

        :param: page: defaulted to 1
        :type: int

        :return: headlines: news api dictionary result
        :rtype: dict
        """
        # TODO: bring back keywords
        headlines = self.newsapi.get_everything(
            #q=self.keywords,
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

        :param: page: defaulted to 1
        :type: int

        :return: headlines: news api dictionary result
        :rtype: dict
        """
        # TODO: bring back keywords
        headlines = self.newsapi.get_top_headlines(
                #q=self.keywords,
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

        :return: sources: all the sources
        :rtype: str
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
        :type: dict
        """
        logging.info(f'dictionary: {dictionary}')
        articles = dictionary.get('articles')
        for article in articles:
            title = article.get('title')
            description = article.get('description')
            publishedAt = article.get('publishedAt')
            row = {'title': title, 'description': description, 'publishedAt': publishedAt}
            logging.info(f'row: {row}')
            self.headlines = self.headlines.append(row, ignore_index=True)
        logging.info(f'updated headlines: {self.headlines}')


    @staticmethod
    def remaining_pages(total_results: int) -> int:
        """
        Get remaining pages to sort through.

        :param total_results: total results returned from news api query
        :type: int

        :return: remaining_pages: remaining pages in query
        :type: int
        """
        return math.ceil((total_results - 100)/100)


    def dataframe_to_csv(self) -> None:
        logging.info('Saving dataframe to csv')
        os.makedirs('outputs', exist_ok=True)
        time.sleep(3)
        self.headlines.to_csv(FILEPATH)


if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.get_all_headlines()
    scraper.get_top_headlines()
    scraper.dataframe_to_csv()