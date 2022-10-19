"""
Utils for models and model evaluation
"""

# ----------- Libraries -----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pickle


# ----------- Functions -----------

def data_split(input_data, output_data, split = 0.2):
    """
    Use train_test_split from sklearn to split data into training and validation sets.

    :param input_data: input or X data

    :param output_data: output or y data

    :param split: train-test split
    :type: double

    :return x_train, x_test, y_train, y_test: the training and validation sets for each of input and output data
    """
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size = split, random_state = 0)
    return x_train, x_test, y_train, y_test


def convergePrices(dataframe: pd.DataFrame, priceLabel: str, scaler = MinMaxScaler()) -> np.ndarray:
    """
    Converge prices to values between 0 and 1.

    :param dataframe: pandas dataframe to obtain price labels from
    :type: pd.DataFrame

    :param priceLabel: column header for price column
    :type: str

    :scaler scaler: scaler to scale with

    :return: scaled_close: cleaned price label column (reshaped to have shape (x, y))
    :rtype: np.ndarray
    """
    close_price = dataframe[priceLabel].values.reshape(-1, 1) # scaler expects data is shaped as (x, y) so we add dummy dimension

    scaled_close = scaler.fit_transform(close_price)

    scaled_close = scaled_close[~np.isnan(scaled_close)] # remove all nan values
    scaled_close = scaled_close.reshape(-1, 1) # reshape after removing nans

    return scaled_close


def saveModel(model, fullyQualifiedFilepath):
    """
    Save model to file.

    :param model: neural network

    :param fullyQualifiedFilepath: file path to save model to
    :type: str
    """
    pickle.dump(model, open(fullyQualifiedFilepath, 'wb'))


def saveScaler(scaler, fullyQualifiedFilepath):
    """
    Save scaler to file.

    :param scaler: scaler to save

    :param fullyQualifiedFilepath: file path to save model to
    :type: str
    """
    pickle.dump(scaler, open(fullyQualifiedFilepath, 'wb'))


def loadModel(fullyQualifiedFilepath):
    """
    Load model from path.

    :param fullyQualifiedFilepath: file path to saved model
    :type: str
    """
    return pickle.load(open(fullyQualifiedFilepath), 'rb')


def loadScaler(fullyQualifiedFilepath):
    """
    Load scaler from path.

    :param fullyQualifiedFilepath: file path to saved scaler
    :type: str
    """
    return pickle.load(open(fullyQualifiedFilepath), 'rb')


def comparisonGraph(y_true, y_pred, coin, output_path):
    """
    Create a graph comparing actual and predicted values.

    :param y_true: actual values
    :type: 1d array-like

    :param y_pred: predicted values
    :type: 1d array-like

    :param coin: name of related coin
    :type: str

    :param output_path: output path to save graph to
    :type: str
    """
    days_passed = len(y_pred)
    time = np.arange(days_passed)
    plt.plot(time, y_true, 'red', label='Actual Price')
    plt.plot(time, y_pred, 'blue', label='Predicted Price')
    plt.legend()
    plt.xlabel('Time[days]')
    plt.ylabel('Price')
    plt.title(f'{coin.capitalize()} price prediction')
    plt.savefig(output_path)