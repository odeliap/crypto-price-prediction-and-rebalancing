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
) -> pd.DataFrame:
    """
    Clean data.

    Parameters
    ----------
    filepath : str
        Path to news file to clean.
    save_columns : list of strings
        Columns to save from the dataset.
    rename_columns_dict : dictionary
        Mapping between existing column names to new column names.

    Returns
    -------
    DataFrame
        Cleaned dataframe.
    """
    dataframe = pd.read_csv(filepath)
    dataframe = dataframe.rename(columns=rename_columns_dict)
    dataframe = dataframe[save_columns]
    return dataframe


def clean_hms_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yyyy-mm-dd hh:mm:ss timestamp to yyyy-mm-dd timestamps.

    Parameters
    ----------
    dataframe : dataframe object
        Dataset for which to convert hms timestamps.

    Returns
    -------
    DataFrame
        Dataset with standardized timestamps
    """
    dataframe['timestamp'] = dataframe['timestamp'].map(lambda x: x[0:x.index(' ')])
    return dataframe


def utc_to_standard_timestamp(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert epoch timestamp to yyyy-mm-dd timestamps

    Parameters
    ----------
    dataframe : dataframe object
        Dataset for which to convert utc timestamps.

    Returns
    -------
    DataFrame
        Dataset with standardized timestamps
    """
    dataframe['timestamp'] = dataframe['timestamp'].map(lambda x: dt.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d"))
    return dataframe
