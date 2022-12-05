"""
LSTM neural network with date input based on historical prices alone.

Used to compare against LSTM with date input based on historical prices and sentiment analysis.

Made by following tutorial:
https://towardsdatascience.com/cryptocurrency-price-prediction-using-lstms-tensorflow-for-hackers-part-iii-264fcdbccd3f
"""

# ------------- Libraries -------------
import logging

import pandas as pd
import numpy as np

from tensorflow import keras
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation
from sklearn.preprocessing import MinMaxScaler

from Utils import converge_prices, save_model, save_scaler, generate_comparison_graph

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

seq_len = 10 # consider changing to 100
dropout = 0.2 # dropout rate
window_size = seq_len - 1 # window size
batch_size = 64 # batch size for processing
test_split = 0.90 # fractional train test split

# Paths for saving model and scaler
modelSavedPath = './outputs/models/PriceLSTMModel'
scalerSavedPath = './outputs/scalers/PriceLSTMScaler'

coins = ['bitcoin', 'ethereum', 'solana'] # available coins

# ------------- Class -------------
class PriceLSTMModel:

    def __init__(
        self,
        coin: str,
        dataframe: pd.DataFrame,
        price_label: str
    ) -> None:
        """
        Initialize PriceLSTMModel.

        Parameters
        ----------
        coin : string
            Coin of interest.
        dataframe : pandas dataframe
            Dataset to process.
        price_label : string
            Column header name for price column.
        """
        self.scaler = MinMaxScaler() # Set scaler to min max scaler
        scaled_price = converge_prices(dataframe, price_label, self.scaler) # Get scaled prices for price column in dataset

        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess(scaled_price, seq_len, test_split) # Split dataset into training and testing data

        self.test_actual_prices = self.scaler.inverse_transform(self.y_test) # Convert test data from scaled to actual prices

        # Setup the model
        self.model = keras.Sequential() # Initialize sequential model
        self.model.add(Bidirectional(
            LSTM(window_size, return_sequences=True),
            input_shape=(window_size, self.X_train.shape[-1])
        )) # Add bidirectional layer to the model
        self.model.add(Dropout(rate=dropout)) # Add dropout rate to the model
        self.model.add(Bidirectional(
            LSTM(window_size, return_sequences=False)
        )) # Add bidirectional layer to the model
        self.model.add(Dense(units=1)) # Add dense layer to the model
        self.model.add(Activation('linear')) # Add activation layer to the model

        self.train() # Train the model

        # Save the model and scaler
        save_model(self.model, f'{modelSavedPath}_{coin}.sav')
        save_scaler(self.scaler, f'{scalerSavedPath}_{coin}.pkl')


    @staticmethod
    def to_sequences(data: pd.DataFrame, seq_len: int) -> np.array:
        """
        Convert data to sequences.

        Parameters
        ----------
        data : pandas dataframe
            Data to transform.
        seq_len : int
            Length for output sequences.

        Returns
        -------
        seq_data : numpy array
            Data converted to sequences.
        """
        d = []

        for index in range(len(data) - seq_len):
            d.append(data[index: index + seq_len])

        return np.array(d)


    def preprocess(
        self,
        raw_data: pd.DataFrame,
        seq_len: int,
        test_split: float
    ) -> (np.array, np.array, np.array, np.array):
        """
        Preprocess data for input to LSTM.

        Parameters
        ----------
        raw_data : pandas dataframe
            Raw data to input.
        seq_len : int
            Length for output sequences.
        test_split : float
            Train-test split.

        Returns
        -------
        X_train : numpy array
            Input training data.
        y_train : numpy array
            Output training data.
        X_test : numpy array
            Input testing data.
        y_test : numpy array
            Output testing data.
        """
        data = self.to_sequences(raw_data, seq_len)

        num_train = int(test_split * data.shape[0])

        X_train = data[:num_train, :-1, :]
        y_train = data[:num_train, -1, :]

        X_test = data[num_train:, :-1, :]
        y_test = data[num_train:, -1, :]

        return X_train, y_train, X_test, y_test


    def train(self) -> None:
        """
        Train the model.
        """
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam'
        ) # compile with mean squared error loss and adam optimiser

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=50,
            batch_size=batch_size,
            shuffle=False,
            validation_split=0.1
        ) # fit training data to model


    def predict(self, X: np.array) -> np.array:
        """
        Predict future prices for given input data.

        Parameters
        ----------
        X : numpy array
            Input data.

        Returns
        -------
        predictions : numpy array
            Predicted future prices.
        """
        scaled_predictions = self.model.predict(X) # Predict scaled prices
        predictions = self.scaler.inverse_transform(scaled_predictions) # Transform scaled predictions to actual prices

        return predictions


def predict(X: np.array, coin: str) -> np.array:
    """
    Predict future prices for given input data.

    Parameters
    ----------
    X : numpy array
        Input data.

    Returns
    -------
    predictions : numpy array
        Predicted future prices.
    """
    # Load model and scaler from saved paths
    model = load_object(f'{modelSavedPath}_{coin}.sav')
    scaler = load_object(f'{scalerSavedPath}_{coin}.pkl')

    scaled_predictions = model.predict(X) # Predict scaled prices
    predictions = scaler.inverse_transform(scaled_predictions) # Transform scaled predictions to actual prices

    return predictions


def main(coin: str, filepath: str) -> None:
    """
    Generate model.

    Show comparison graph of actual prices compared to predicted prices.

    Parameters
    ----------
    coin : string
        Name of coin to get inputs for.
    filepath : string
        Path to csv file with related data for coin.
    """
    logging.info(f"starting up {coin} price lstm model")

    dataframe = pd.read_csv(filepath, parse_dates=['timestamp']) # Read in dataset
    dataframe = dataframe.sort_values('timestamp') # Sort dataset by timestamp

    model = PriceLSTMModel(coin, dataframe, 'open') # Create a PriceLSTMModel for the open price
    predictions = model.predict(model.X_test) # Get model's predictions for the test input data

    # Report predictions vs actual prices
    logging.info("Predictions:")
    logging.info(f'{predictions}\n')
    logging.info("Actual Prices:")
    logging.info(f'{model.test_actual_prices}\n')

    # Graph predicted prices against actual prices
    generate_comparison_graph(model.test_actual_prices, predictions, coin, f'outputs/graphs/PriceLSTMModel_comparison_{coin}.png')


if __name__ == "__main__":
    """
    Loop over available coins to generate a saved model and metrics for each.
    """
    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)