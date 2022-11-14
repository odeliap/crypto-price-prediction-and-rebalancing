"""
Format datasets for input to target models for evaluation
"""

# ------------- Libraries -------------

import pandas as pd
from typing import List


# ------------- Constants -------------

# List of all the sentiment columns
sentiment_cols = ['subjectivity', 'polarity', 'compound', 'negative', 'neutral', 'positive']

# Coins available
coins = ['bitcoin', 'ethereum', 'solana']


# ------------- Functions -------------

def replaceWithConstantSentiment(dataframe: pd.DataFrame, value: int, sentiment_cols: List[str]):
    """
    Replace all sentiment columns with constant values.

    Parameters
    ----------
    dataframe : dataframe object
        Dataframe to alter.
    value : int
        Value to set sentiment columns to.
    sentiment_cols : list of strings
        Sentiment columns for which to alter values.

    Returns
    -------
    DataFrame
        Altered dataframe.
    """
    for col in sentiment_cols:
        dataframe[col] = value

    return dataframe


if __name__ == "__main__":
    """
    Create constant-sentiment datasets
    
    Loops over all coins available and generates a constant-sentiment dataset for each coin's
    sentiment dataset (under the sentiment analysis outputs)
    """
    for coin in coins:
        dataframe = pd.read_csv(f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv', index_col=False)
        dataframe = dataframe.iloc[:, 1:]
        constantSentimentDataframe = replaceWithConstantSentiment(dataframe, 1, sentiment_cols)
        constantSentimentDataframe.to_csv(f'datasets/{coin}_constant_sentiment_dataset.csv')