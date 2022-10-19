"""
LSTM neural network with date input based on historical prices (solely).

Used to compare against LSTM with date input based on historical prices and sentiment analysis.

Made by following tutorial:
https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f
"""

# ------------- Libraries -------------
import logging

from tensorflow import keras
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from Utils import convergePrices, saveModel, saveScaler, loadModel, loadScaler, comparisonGraph

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

SEQ_LEN = 10 # consider changing to 100
DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1
BATCH_SIZE = 64
TRAIN_SPLIT = 0.90

modelSavedPath = './outputs/models/PriceLSTMModel'
scalerSavedPath = './outputs/scalers/PriceLSTMScaler'

# ------------- Class -------------
class PriceLSTMModel:

    def __init__(self, coin: str, dataframe: pd.DataFrame, priceLabel: str):
        """
        Initialize PriceLSTMModel

        :param coin: coin of interest
        :type: str

        :param dataframe: pandas dataframe to process
        :type: pd.DataFrame

        :param priceLabel: column header for price column
        :type: str
        """
        self.scaler = MinMaxScaler()
        scaled_price = convergePrices(dataframe, priceLabel, self.scaler)

        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess(scaled_price, SEQ_LEN, TRAIN_SPLIT)

        self.test_actual_prices = self.scaler.inverse_transform(self.y_test)

        self.model = keras.Sequential()

        self.model.add(Bidirectional(
            LSTM(WINDOW_SIZE, return_sequences=True),
            input_shape=(WINDOW_SIZE, self.X_train.shape[-1])
        ))
        self.model.add(Dropout(rate=DROPOUT))

        self.model.add(Bidirectional(
            LSTM(WINDOW_SIZE, return_sequences=False)
        ))

        self.model.add(Dense(units=1))
        self.model.add(Activation('linear'))
        self.train()
        saveScaler(self.scaler, f'{scalerSavedPath}_{coin}.pkl')
        saveModel(self.model, f'{modelSavedPath}_{coin}.sav')


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

        num_train = int(train_split * data.shape[0])

        X_train = data[:num_train, :-1, :]
        y_train = data[:num_train, -1, :]

        X_test = data[num_train:, :-1, :]
        y_test = data[num_train:, -1, :]

        return X_train, y_train, X_test, y_test


    def train(self):
        """
        Train the model.
        """
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam'
        )

        self.history = self.model.fit(
            self.X_train,
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


def predict(input, coin):
    """
    Make predictions.

    :param input: input or x data

    :param coin: coin of interest

    :return predictions: output predictions
    """
    model = loadModel(f'{modelSavedPath}_{coin}.sav')
    scaled_predictions = model.predict(input)
    scaler = loadScaler(f'{scalerSavedPath}_{coin}.pkl')
    predictions = scaler.inverse_transform(scaled_predictions)
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

    dataframe = pd.read_csv(filepath, parse_dates=['timestamp'])
    dataframe = dataframe.sort_values('timestamp')

    model = PriceLSTMModel(coin, dataframe, 'open')
    predictions = model.predict(model.X_test)

    logging.info("PREDICTIONS:")
    logging.info(f'{predictions}\n')
    logging.info("ACTUAL PRICES:")
    logging.info(f'{model.test_actual_prices}\n')

    comparisonGraph(model.test_actual_prices, predictions, coin, f'outputs/graphs/PriceLSTMModel_comparison_{coin}.png')


if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)