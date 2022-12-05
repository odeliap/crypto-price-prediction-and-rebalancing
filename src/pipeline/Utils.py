"""
Utils for end-to-end pipeline
"""

# ----------- Libraries -----------

import pandas as pd

import os
import shutil

from typing import List

from datetime import datetime, timedelta

# ----------- Functions -----------

def make_directory(path: str) -> None:
    """
    Make directory.

    Parameters
    __________
    path : string
        Full path name for directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def make_dirs(directories: List[str]) -> None:
    """
    Make directories.

    Parameters
    __________
    directories : list of strings
        Directories to create.
    """
    for dir in directories:
        make_directory(dir)


def delete_dirs(directories: List[str]) -> None:
    """
    Delete directories.

    Parameters
    __________
    directories : list of strings
        Directories to delete.
    """
    for dir in directories:
        shutil.rmtree(dir)


def clean_prices_dataframe(
        prev_prices: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        date_format
    ) -> pd.DataFrame:
    """
    Clean prices dataframe.

    Parameters
    __________
    prev_prices : pandas dataframe
        Uncleaned previous prices.
    start_date : datetime
        Date to start scraping historical data from.
    end_date : datetime
        Date to stop scraping historical data to.
    date_format : str
        Date format for formatting datetime objects.

    Returns
    _______
    prev_prices_clean : pandas dataframe
        Cleaned dataframe of previous prices.
    """
    prev_prices_clean = prev_prices.dropna()
    rename_dict = dict()
    for col in prev_prices_clean.columns:
        col_lowercase = col.lower()
        rename_dict[col] = col_lowercase
    prev_prices_clean = prev_prices_clean.rename(rename_dict, axis=1)
    timestamps = pd.date_range(start_date + timedelta(days=1), end_date, freq='d').tolist()
    timestamps_str = [(lambda x: x.strftime(date_format))(x) for x in timestamps]
    prev_prices_clean['timestamp'] = timestamps_str
    return prev_prices_clean


def combine_dataframes(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """
    Combine price and news dataframes.

    Parameters
    __________
    prices : pandas dataframe
        Historical prices.
    news : pandas dataframe
        Corresponding historical news.

    Returns
    _______
    combined_dataframe : pandas dataframe
        Combined price and news dataframe.
    """
    news['text'] = news.groupby(['timestamp'])['text'].transform(
        lambda x: ' '.join(map(str, x)))  # Group by timestamp to combine all the news into a single text column
    news = news.drop_duplicates(subset='timestamp', keep='first')  # Drop duplicate timestamp entries
    combined_dataframe = pd.merge(prices, news, on='timestamp', how='left').fillna("neutral")
    return combined_dataframe


def format_sentiment_input_for_predictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Format sentiment analysis output for input to price prediction model.

    Parameters
    __________
    dataframe : pandas dataframe
        Dataframe to transform for input.

    Returns
    _______
    input : pandas dataframe
        Formatted dataframe.
    """
    dataframe = dataframe.drop(dataframe.columns[[0]], axis=1)
    if 'open' in dataframe.columns:
        dataframe = dataframe.drop(columns=['open'])
    return dataframe