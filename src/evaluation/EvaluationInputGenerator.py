"""
Functions to format datasets for input to models for evaluation
"""

# ------------- Libraries -------------

import pandas as pd


# ------------- Methods -------------

def replaceWithConstantSentiment(dataframe: pd.DataFrame, value: int):
    """
    Replace all sentiment columns with constant values.

    :param dataframe: dataframe to alter
    :type: pd.DataFrame

    :param value: value to set sentiment columns to
    :type: int

    :return dataframe: altered dataframe
    :rtype: pd.DataFrame
    """
    sentiment_cols = ['subjectivity', 'polarity', 'compound', 'negative', 'neutral', 'positive']

    for col in sentiment_cols:
        dataframe[col] = value

    return dataframe


if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        dataframe = pd.read_csv(f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv', index_col=False)
        dataframe = dataframe.iloc[:, 1:]
        constantSentimentDataframe = replaceWithConstantSentiment(dataframe, 1)
        constantSentimentDataframe.to_csv(f'datasets/{coin}_constant_sentiment_dataset.csv')