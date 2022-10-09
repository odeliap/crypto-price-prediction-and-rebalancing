"""
Sample non-LSTM sentiment model (predicts price increase/decrease only)

Closely modeled after tutorial model (Spring Seminar tutorial)
"""

# ----------- Libraries -----------
import logging

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

import numpy as np

from Utils import data_split, saveModel, loadModel, comparisonGraph

# Set logging level
logging.basicConfig(level=logging.INFO)

# ----------- Constants -----------

modelSavedPath = './outputs/SampleSentimentModel.sav'

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
        drop_columns = ['open', 'high', 'low']
        features = dataframe
        features = np.array(features.drop(drop_columns, axis=1))
        price_data = np.array(dataframe[priceLabel])

        label_price_data = []

        previous_price = 0
        for price in price_data:
            if price > previous_price:
                label_price_data.append(1)
            else:
                label_price_data.append(0)
            previous_price = price

        self.x_train, self.x_test, self.y_train, self.y_test = data_split(features, label_price_data)

        self.model = LinearDiscriminantAnalysis().fit(self.x_train, self.y_train)
        saveModel(self.model, modelSavedPath)


    def predict(self, input_data):
        """
        Predict price for input data.

        :param input_data: input or x data

        :return predictions: predicted prices
        """
        predictions = self.model.predict(input_data)
        return predictions


def predict(input_data):
    """
    Predict price for input data based on saved model.

    :param input_data: input or x data

    :return predictions: predicted prices
    """
    loaded_model = loadModel(modelSavedPath)
    predictions = loaded_model.predict(input_data)
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
    logging.info(classification_report(model.y_test, predictions))
    comparisonGraph(model.y_test, predictions, coin)


if __name__ == "__main__":
    """
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)
    """
    filepath = f'../sentiment_analysis/outputs/sample_sentiment_dataset.csv'
    main('bitcoin', filepath)