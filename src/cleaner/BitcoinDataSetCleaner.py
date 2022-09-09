"""
Dataset Combiner
"""

# ------------- Libraries -------------
import logging

from os import walk

import pandas as pd

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

INPUT_DATASET = '../../datasets/news/unprocessed/Bitcoin_tweets.csv'
OUTPUT_FILEPATH = '../../datasets/news/clean/Bitcoin_tweets.csv'

# ------------- Runner -------------

if __name__ == "__main__":
    save_headers = ['date', 'text']
    dataframe = pd.read_csv(INPUT_DATASET)
    clean_dataframe = dataframe[save_headers]
    timestamps = clean_dataframe['date']
    clean_timestamps = []
    for timestamp in timestamps:
        clean_timestamp = timestamp[0:timestamp.index(' ')]
        clean_timestamps.append(clean_timestamp)
    clean_dataframe['date'] = clean_timestamps
    clean_dataframe = clean_dataframe.rename(columns={'date': 'timestamp'})
    clean_dataframe['coin'] = 'Bitcoin'
    clean_dataframe.to_csv(OUTPUT_FILEPATH)

