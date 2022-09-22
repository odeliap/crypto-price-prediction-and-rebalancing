"""
Class to process collected solana datasets into cohesive dataset
"""

# ------------- Libraries -------------
import logging

import pandas as pd

from Utils import clean_data, utc_to_standard_timestamp, clean_hms_timestamps

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

news_filepath = 'datasets/news/solana/solana_news.csv'
price_filepath = 'datasets/price/solana/solana_price.csv'

# ------------- Class -------------
class SolanaProcessor:

    def __init__(self):
        """
        Initialize SolanaProcessor
        """
        self.news_dataframe = utc_to_standard_timestamp(
            clean_data(
                news_filepath,
                save_columns=['timestamp', 'text'],
                rename_columns_dict={'created_utc': 'timestamp', 'body': 'text'}
            )
        )

        self.price_dataframe = clean_hms_timestamps(
            clean_data(
                price_filepath,
                save_columns=['timestamp', 'open', 'high', 'low'],
                rename_columns_dict={'Date': 'timestamp', 'High': 'high', 'Low': 'low', 'Open': 'open',
                                     'Close': 'close'}
            )
        )


    def combine_dataframes(self):
        """
        Combine news and price dataframes into cohesive dataframe.

        :return: dataframe: clean combined dataframe
        :rtype pd.DataFrame
        """
        self.news_dataframe['text'] = self.news_dataframe['text'].transform(lambda x: x.replace(',', ' ').replace('\n', ' '))
        combined_dataframe = pd.merge(self.price_dataframe, self.news_dataframe, on='timestamp', how='inner')

        combined_dataframe['text'] = combined_dataframe.groupby(['timestamp'])['text'].transform(
            lambda x: ' '.join(map(str, x)))
        combined_dataframe.dropna(inplace=True)

        combined_dataframe = combined_dataframe.drop_duplicates(subset='timestamp', keep='first')
        return combined_dataframe



if __name__ == "__main__":
    processor = SolanaProcessor()
    dataframe = processor.combine_dataframes()
    dataframe.to_csv('outputs/solana_dataset.csv', index=None)