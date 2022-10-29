"""
Class to process collected bitcoin datasets into cohesive dataset
"""

# ------------- Libraries -------------
import logging

import pandas as pd

from typing import List

from Utils import clean_data

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

news_filepath = 'datasets/news/bitcoin/bitcoin_news.csv'
news_and_price_filepath1 = 'datasets/news_and_price/bitcoin/bitcoin_news_and_price1.csv'
news_and_price_filepath2 = 'datasets/news_and_price/bitcoin/bitcoin_news_and_price2.csv'
price_filepath = 'datasets/price/bitcoin/bitcoin_price.csv'


# ------------- Class -------------
class BitcoinProcessor:

    def __init__(self):
        """
        Initialize BitcoinProcessor
        """
        news_dataframe = clean_data(filepath = news_filepath, save_columns = ['timestamp', 'text'], rename_columns_dict = {'Date': 'timestamp', 'Title': 'text'})
        price_dataframe = clean_data(filepath = price_filepath, save_columns = ['timestamp', 'open', 'high', 'low'], rename_columns_dict = {'Date': 'timestamp', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Close': 'close'})
        news_and_price_dataframe1 = self.clean_news_and_price(news_and_price_filepath1)
        news_and_price_dataframe2 = self.clean_news_and_price(news_and_price_filepath2)

        self.dataframes = [news_dataframe, news_and_price_dataframe1, news_and_price_dataframe2, price_dataframe]


    @staticmethod
    def combine_dataframes_with_transform(dataframes: List[pd.DataFrame]):
        """
        Combine news and price dataframes into cohesive dataframe

        :param dataframes: list of pd.DataFrame objects to combine

        :return: dataframe: clean combined dataframe
        :rtype pd.DataFrame
        """
        combined_dataframe = pd.concat(dataframes, ignore_index=True)
        combined_dataframe['text'] = combined_dataframe.groupby(['timestamp'])['text'].transform(
            lambda x: ' '.join(map(str, x)))
        combined_dataframe.dropna(inplace=True)
        return combined_dataframe


    @staticmethod
    def clean_news_and_price(filepath):
        """
        Clean news and price file.

        :return: dataframe: cleaned dataframe
        :rtype pd.DataFrame
        """
        rename_columns_dict = {'date': 'timestamp'}
        save_columns = ['timestamp', 'text', 'price', 'open', 'high', 'low']

        dataframe = pd.read_csv(filepath)
        dataframe = dataframe.rename(columns=rename_columns_dict)

        columns = dataframe.columns.tolist()
        news_columns = []

        for col in columns:
            if col.startswith('top'):
                news_columns.append(col)

        dataframe['text'] = dataframe[news_columns].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
        )
        dataframe = dataframe[save_columns]

        return dataframe


if __name__ == "__main__":
    processor = BitcoinProcessor()
    dataframe = processor.combine_dataframes_with_transform(processor.dataframes)
    dataframe.to_csv('outputs/bitcoin_dataset.csv', index=None)