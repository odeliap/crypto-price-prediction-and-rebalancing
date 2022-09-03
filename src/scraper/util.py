"""
Utils for scraper package
"""

def merge(dictionary1: dict, dictionary2: dict):
    """
    Merges two dictionaries into single dictionary.

    :param dictionary1: first dictionary
    :type: dict

    :param dictionary2: second dictionary
    :type: dict

    :return: Python dictionary.
    :rtype: dict
    """
    result = {**dictionary1, **dictionary2}
    return result