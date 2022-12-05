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

def data_split(
    X: np.array,
    y: np.array,
    split = 0.2
) -> (np.array, np.array, np.array, np.array):
    """
    Use train_test_split from sklearn to split data into training and validation sets.

    Parameters
    __________
    X : array
        Input data.
    y : array
        Corresponding output data.
    test_split : float
        Train-test split.

    Returns
    _______
    x_train : array
        Training input data.
    x_test : array
        Testing input data.
    y_train : array
        Training output data.
    y_test : array
        Testing output data.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = split, random_state = 0)
    return x_train, x_test, y_train, y_test


def converge_prices(
    dataframe: pd.DataFrame,
    price_label: str,
    scaler = MinMaxScaler()
) -> np.ndarray:
    """
    Converge prices to values between 0 and 1.

    Parameters
    __________
    dataframe : pandas dataframe
        Dataset from which to obtain price labels.
    price_label : string
        Column header name for price column.
    scaler : scaler
        Scaler with which to scale prices.

    Returns
    _______
    scaled_close : 2d array-like
        Cleaned price label column (reshaped to have shape (x, y))
    """
    close_price = dataframe[price_label].values.reshape(-1, 1) # scaler expects data is shaped as (x, y) so we add dummy dimension

    scaled_close = scaler.fit_transform(close_price)

    scaled_close = scaled_close[~np.isnan(scaled_close)] # remove all nan values
    scaled_close = scaled_close.reshape(-1, 1) # reshape after removing nans

    return scaled_close


def save_model(model, fully_qualified_filepath: str) -> None:
    """
    Save model to file.

    Parameters
    __________
    model : model
        Neural network.
    fully_qualified_filepath : string
        File path to save model to.
    """
    pickle.dump(model, open(fully_qualified_filepath, 'wb'))


def save_scaler(scaler, fully_qualified_filepath: str) -> None:
    """
    Save scaler to file.

    Parameters
    __________
    scaler : scaler
        Scaler.
    fully_qualified_filepath : string
        File path to save scaler to.
    """
    pickle.dump(scaler, open(fully_qualified_filepath, 'wb'))


def generate_comparison_graph(
    y_true: np.array,
    y_pred: np.array,
    coin: str,
    output_path: str
) -> None:
    """
    Create a graph comparing actual and predicted values.

    Parameters
    __________
    y_true : 1d array-like
        Actual price values.
    y_pred : 1d array-like
        Predicted price values.
    coin : string
        Name of related coin.
    output_path : string
        Output path to save graph to.
    """
    days_passed = len(y_pred)
    time = np.arange(days_passed)
    plt.style.use('seaborn-v0_8-pastel')
    plt.figure(figsize=(10, 6))  # plotting
    plt.plot(time, y_true, label='Actual Price')
    plt.plot(time, y_pred, label='LSTM Predicted Price')
    plt.legend()
    plt.xlabel('Time[days]')
    plt.ylabel('Price (USD)')
    plt.title(f'{coin.capitalize()} price prediction')
    plt.savefig(output_path)
    plt.show()