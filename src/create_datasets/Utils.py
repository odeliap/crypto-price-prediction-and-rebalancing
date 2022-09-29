"""
Utils for creating datasets
"""

# ------------- Libraries -------------
import logging

import pandas as pd
from typing import List

import datetime as dt

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Functions -------------

def clean_data(
        filepath: str,
        save_columns: List[str],
        rename_columns_dict: dict
):
    """
    Clean data.

    :param filepath: path to news file to clean
    :type: str

    :param save_columns: list of columns to save
    :type: List[str]

    :param rename_columns_dict: dictionary mapping existing column names to new column names
    :type: dict

    :return: dataframe: cleaned dataframe
    :rtype pd.DataFrame
    """
    dataframe = pd.read_csv(filepath)
    dataframe = dataframe.rename(columns=rename_columns_dict)
    dataframe = dataframe[save_columns]
    return dataframe


def clean_hms_timestamps(dataframe: pd.DataFrame):
    """
    Convert yyyy-mm-dd hh:mm:ss timestamp to yyyy-mm-dd timestamps

    :param dataframe: dataframe to clean timestamps for
    :type: pd.DataFrame

    :return: dataframe: dataframe with cleaned timestamps
    :rtype pd.DataFrame
    """
    dataframe['timestamp'] = dataframe['timestamp'].map(lambda x: x[0:x.index(' ')])
    return dataframe


def utc_to_standard_timestamp(dataframe: pd.DataFrame):
    """
    Convert epoch timestamp to yyyy-mm-dd timestamps

    :param dataframe: dataframe to clean timestamps for
    :type: pd.DataFrame

    :return: dataframe: dataframe with cleaned timestamps
    :rtype pd.DataFrame
    """
    dataframe['timestamp'] = dataframe['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d"))
    return dataframe