"""
Utils for scraper package
"""

# ------------- Libraries -------------
import logging

import time

import os

# Set logging level
logging.basicConfig(level=logging.INFO)

def merge(dictionary1: dict, dictionary2: dict) -> dict:
    """
    Merges two dictionaries into single dictionary.

    Parameters
    __________
    dictionary1 : dictionary object
        First dictionary to merge.
    dictionary2 : dictionary object
        Second dictionary to merge.

    Returns
    _______
    dictionary : dictionary object
        Result of merging the two dictionaries.
    """
    dictionary = {**dictionary1, **dictionary2}
    return dictionary


def dataframe_to_csv(dataframe, filepath) -> None:
    """
    Save dataframe as csv file to datasets directory.

    Parameters
    __________
    dataframe : pandas dataframe
        Dataframe to save as a csv.
    filepath : string
        Filepath to where to save dataframe to.
    """
    logging.info('Saving dataframe to csv')
    os.makedirs('datasets', exist_ok=True)
    time.sleep(3)
    dataframe.to_csv(filepath, index=None)