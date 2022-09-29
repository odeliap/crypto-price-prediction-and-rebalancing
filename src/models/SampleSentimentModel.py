"""
Sample non-LSTM sentiment model

Closely modeled after tutorial model (Spring Seminar tutorial)
"""

# ----------- Libraries -----------
import logging

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from Utils import data_split

# Set logging level
logging.basicConfig(level=logging.INFO)

# ----------- Constants -----------



# ----------- Class -----------

class SampleSentimentModel:

    def __init__(self, dataframe: pd.DataFrame, priceLabel: str):
        """
        Initialize SampleSentimentModel

        :param dataframe: pandas dataframe to process
        :type: pd.DataFrame

        :param priceLabel: column header for price column
        :type: str
        """
        features = dataframe['subjectivity', 'polarity', 'compound', 'negative', 'neutral', 'positive']
        price_data = dataframe[priceLabel]

        self.x_train, self.y_train, self.x_test, self.y_test = data_split(features, price_data)

        self.model = LinearDiscriminantAnalysis().fit(self.x_train, self.y_train)


    def predict(self, input_data):
        """
        Predict price for input data.

        :param input_data: input or x data

        :return predictions: predicted prices
        """
        predictions = self.model.predict(input_data)
        return predictions


def main(coin: str, filepath: str):
    """
    Generate model

    :param coin: name of coin to get inputs for
    :type: str

    :param filepath: path to csv file with related data for coin
    :type: str

    :return predictions: predicted prices
    """
    logging.info(f"starting up {coin} sample sentiment model")

    dataframe = pd.read_csv(filepath)

    model = SampleSentimentModel(dataframe, 'open')
    predictions = model.predict(model.x_test)
    return predictions


# TODO: add method to compare predictions against actual prices y_test