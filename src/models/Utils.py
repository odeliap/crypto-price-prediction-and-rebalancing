"""
Utils for models and model evaluation
"""

# ----------- Libraries -----------
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ----------- Functions -----------

def data_split(input_data, output_data, split = 0.2):
    """
    Use train_test_split from sklearn to split data into training and validation sets.

    :param input_data: input or X data

    :param output_data: output or y data

    :param split: train-test split
    :type: double

    :return x_train, y_train, x_test, y_test: the training and validation sets for each of input and output data
    """
    x_train, y_train, x_test, y_test = train_test_split(input_data, output_data, test_size = split, random_state = 0)
    return x_train, y_train, x_test, y_test


def convergePrices(dataframe: pd.DataFrame, priceLabel: str) -> np.ndarray:
    """
    Converge prices to values between 0 and 1.

    :param dataframe: pandas dataframe to obtain price labels from
    :type: pd.DataFrame

    :param priceLabel: column header for price column
    :type: str

    :return: scaled_close: cleaned price label column (reshaped to have shape (x, y))
    :rtype: np.ndarray
    """
    scaler = MinMaxScaler()
    close_price = dataframe[priceLabel].values.reshape(-1, 1) # scaler expects data is shaped as (x, y) so we add dummy dimension

    scaled_close = scaler.fit_transform(close_price)

    scaled_close = scaled_close[~np.isnan(scaled_close)] # remove all nan values
    scaled_close = scaled_close.reshape(-1, 1) # reshape after removing nans

    return scaled_close