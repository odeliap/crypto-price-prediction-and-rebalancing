"""
Utils for scraper package
"""

# ------------- Libraries -------------
import logging

import time

import os

# Set logging level
logging.basicConfig(level=logging.INFO)

def merge(dictionary1: dict, dictionary2: dict):
    """
    Merges two dictionaries into single dictionary.

    :param dictionary1: first dictionary
    :type: dict

    :param dictionary2: second dictionary
    :type: dict

    :return Python dictionary.
    :rtype dict
    """
    result = {**dictionary1, **dictionary2}
    return result


def dataframe_to_csv(dataframe, filepath) -> None:
    """
    Save dataframe as csv file to datasets directory.

    :param dataframe: pandas dataframe to save as csv
    :type: pd.DataFrame

    :param filepath: filepath where to save dataframe to
    :type: str
    """
    logging.info('Saving dataframe to csv')
    os.makedirs('datasets', exist_ok=True)
    time.sleep(3)
    dataframe.to_csv(filepath, index=None)