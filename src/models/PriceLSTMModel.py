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
from sklearn.model_selection import train_test_split

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

SEQ_LEN = 100
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1
BATCH_SIZE = 64

# ------------- Class -------------
class PriceLSTMModel:

    def __init__(self, dataframe: pd.DataFrame, priceLabel: str):
        """
        Initialize PriceLSTMModel

        :param dataframe: pandas dataframe to process
        :type pd.DataFrame

        :param priceLabel: column header for price column
        :type str
        """
        scaled_close = self.convergePrices(dataframe, priceLabel)

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


    @staticmethod
    def convergePrices(dataframe: pd.DataFrame, priceLabel: str) -> np.ndarray:
        """
        Converge prices to values between 0 and 1.

        :param dataframe: pandas dataframe to obtain price labels from
        :type pd.DataFrame

        :param priceLabel: column header for price column
        :type str

        :return: scaled_close: cleaned price label column (reshaped to have shape (x, y))
        :rtype: np.ndarray
        """
        scaler = MinMaxScaler()

        close_price = dataframe[priceLabel].values.reshape(-1, 1) # scaler expects data is shaped as (x, y) so we add dummy dimension

        scaled_close = scaler.fit_transform(close_price)

        scaled_close = scaled_close[~np.isnan(scaled_close)] # remove all nan values
        scaled_close = scaled_close.reshape(-1, 1) # reshape after removing nans

        return scaled_close

    @staticmethod
    def to_sequences(data, seq_len) -> np.array:
        """
        Convert to sequences.

        :param data:
        :type

        :param seq_len:
        :type

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
        :type

        :param seq_len:
        :type

        :return x_train, y_train, x_test, y_test
        """
        data = self.to_sequences(raw_data, seq_len)

        X = data[:, :-1, :]
        Y = data[:, -1, :]

        x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=train_split, random_state=0)

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


if __name__ == "__main__":
    logging.info("starting up LSTM model")
