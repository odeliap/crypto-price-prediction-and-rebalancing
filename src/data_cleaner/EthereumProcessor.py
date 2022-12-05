"""
Process collected ethereum datasets into clean cohesive dataset
"""

# ------------- Libraries -------------
import logging

import pandas as pd

from Utils import clean_data, clean_hms_timestamps

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

# Set filepaths to input datasets to be combined and cleaned
news_filepath = 'datasets/news/ethereum/ethereum_news.csv'
price_filepath1 = 'datasets/price/ethereum/ethereum_price1.csv'
price_filepath2 = 'datasets/price/ethereum/ethereum_price2.csv'


# ------------- Class -------------
class EthereumProcessor:

    def __init__(
        self
    ) -> None:
        """
        Initialize EthereumProcessor
        """
        price_save_columns = ['timestamp', 'open', 'high', 'low', 'close']  # Columns to save for the price datasets

        # Clean the datasets
        self.news_dataframe = clean_data(
            news_filepath,
            save_columns=['timestamp', 'text'],
            rename_columns_dict={'user_created': 'timestamp'}
        )
        self.price_dataframe1 = clean_hms_timestamps(
            clean_data(
                price_filepath1,
                price_save_columns,
                rename_columns_dict={
                    'Open Time': 'timestamp',
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Close': 'close'
                }
            )
        )
        self.price_dataframe2 = clean_hms_timestamps(
            clean_data(
                price_filepath2,
                price_save_columns,
                rename_columns_dict={
                    'Date': 'timestamp',
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Close': 'close'
                }
            )
        )

    def combine_dataframes(self) -> pd.DataFrame:
        """
        Combine news and price dataframes into cohesive dataframe.

        Returns
        -------
        DataFrame
            Clean combined dataframe.
        """
        combined_price = pd.concat(
            [self.price_dataframe1, self.price_dataframe2],
            ignore_index=True
        )  # Concatenate the price dataframes
        # Get the mean price across the price dataframes (grouped by timestamp)
        combined_price = combined_price.groupby('timestamp').mean()

        # Combine the news and price dataframes
        combined_dataframe = pd.merge(combined_price, self.news_dataframe, on='timestamp', how='inner')

        # Group by timestamp to combine all the news into a text column
        combined_dataframe['text'] = combined_dataframe.groupby(['timestamp'])['text'].transform(
            lambda x: ' '.join(map(str, x)))
        combined_dataframe.dropna(inplace=True)
        return combined_dataframe


if __name__ == "__main__":
    """
    Initialize EthereumProcessor

    Create a combined, clean dataset and save it to the outputs directory.
    """
    processor = EthereumProcessor()
    dataframe = processor.combine_dataframes()
    dataframe.to_csv('outputs/ethereum_dataset.csv', index=False)
