"""
LSTM neural network with date input based on historical prices (solely).

Used to compare against LSTM with date input based on historical prices and sentiment analysis.

Made by following tutorial:
https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f
"""

# ------------- Libraries -------------
import logging

from tensorflow import keras
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, Dense, Activation

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from Utils import data_split, convergePrices, saveModel, loadModel, comparisonGraph

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

SEQ_LEN = 100
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1
BATCH_SIZE = 64

modelSavedPath = './outputs/PriceLSTMModel.sav'

# ------------- Class -------------
class PriceLSTMModel:

    def __init__(self, dataframe: pd.DataFrame, priceLabel: str):
        """
        Initialize PriceLSTMModel

        :param dataframe: pandas dataframe to process
        :type: pd.DataFrame

        :param priceLabel: column header for price column
        :type: str
        """
        self.scaler = MinMaxScaler()
        scaled_close = convergePrices(dataframe, priceLabel)

        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess(scaled_close, SEQ_LEN, 0.95)

        self.model = keras.Sequential()

        self.model.add(Bidirectional(
            CuDNNLSTM(WINDOW_SIZE, return_sequences=True),
            input_shape=(WINDOW_SIZE, self.x_train.shape[-1])
        ))
        self.model.add(Dropout(rate=DROPOUT))

        self.model.add(Bidirectional(
            CuDNNLSTM(WINDOW_SIZE, return_sequences=False)
        ))

        self.model.add(Dense(units=1))
        self.model.add(Activation('linear'))
        saveModel(self.model, modelSavedPath)


    @staticmethod
    def to_sequences(data, seq_len) -> np.array:
        """
        Convert to sequences.

        :param data

        :param seq_len

        :return seq_data: data converted to sequence
        :rtype np.array
        """
        d = []

        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])

        return np.array(d)


    def preprocess(self, raw_data, seq_len, train_split):
        """
        Preprocess data for input to LSTM.

        :param raw_data: raw data to input

        :param seq_len

        :param train_split: train-test split divide

        :return x_train, y_train, x_test, y_test
        """
        data = self.to_sequences(raw_data, seq_len)

        X = data[:, :-1, :]
        Y = data[:, -1, :]

        x_train, x_test, y_train, y_test = data_split(X, Y, train_split)

        return x_train, y_train, x_test, y_test


    def train(self):
        """
        Train the model.
        """
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam'
        )

        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=50,
            batch_size=BATCH_SIZE,
            shuffle=False,
            validation_split=0.1
        )

    def predict(self, input_data):
        """
        Predict price for given input data.

        :param input_data: input or x data

        :return prediction: predicted price
        """
        scaled_predictions = self.model.predict(input_data)
        predictions = self.scaler.inverse_transform(scaled_predictions)

        return predictions


def predict(input_data):
    """
    Predict price for input data based on saved model.

    :param input_data: input or x data

    :return predictions: predicted prices
    """
    loaded_model = loadModel(modelSavedPath)
    scaled_predictions = loaded_model.predict(input_data)
    predictions = MinMaxScaler().inverse_transform(scaled_predictions)
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
    logging.info(f"starting up {coin} price lstm model")

    dataframe = pd.read_csv(filepath)

    model = PriceLSTMModel(dataframe, 'open')
    predictions = model.predict(model.x_test)
    comparisonGraph(model.y_test, predictions, coin)


if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)