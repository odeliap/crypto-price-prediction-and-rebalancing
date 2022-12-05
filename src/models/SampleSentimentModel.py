"""
Sample non-LSTM sentiment model.

Predicts price increase/decrease only using Linear Discriminant Analysis.

Closely modeled after tutorial model from Spring Seminar tutorial assignment.
"""

# ----------- Libraries -----------
import logging

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

import numpy as np

from Utils import data_split, save_model
from Unpickler import load_object

# Set logging level
logging.basicConfig(level=logging.INFO)

# ----------- Constants -----------

modelSavedPath = './outputs/models/SampleSentimentModel'  # Path for saving model

test_split = 0.1  # train test split

coins = ['bitcoin', 'ethereum', 'solana']  # available coins

# ----------- Class -----------


class SampleSentimentModel:

    def __init__(
        self,
        coin_name: str,
        dataframe: pd.DataFrame,
        price_label: str
    ) -> None:
        """
        Initialize SampleSentimentModel

        Parameters
        ----------
        coin_name : string
            Coin of interest.
        dataframe : pandas dataframe
            Dataset to process.
        price_label : string
            Column header name for price column.
        """
        drop_columns = ['open', 'high', 'low', 'timestamp']  # Define columns to drop from the dataframe
        features = dataframe
        features = np.array(features.drop(drop_columns, axis=1))  # Create input features from non drop columns
        price_data = np.array(dataframe[price_label])  # Define price output data

        label_price_data = []  # Create list to store label price data

        # Convert prices to labels (0 = decreasing, 1 = increasing)
        previous_price = 0
        for price in price_data:
            if price > previous_price:
                label_price_data.append(1)
            else:
                label_price_data.append(0)
            previous_price = price

        # Split input data into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = data_split(features, label_price_data, test_split)

        # Create a linear discriminant model and fit the training data to it
        self.model = LinearDiscriminantAnalysis().fit(self.x_train, self.y_train)

        # Save the model
        save_model(self.model, f'{modelSavedPath}_{coin_name}.sav')

    def predict(self, x: np.array) -> np.array:
        """
        Predict future prices from input data.

        Parameters
        ----------
        x : numpy array
            Input data.

        Returns
        -------
        predictions : numpy array
            Predicted future prices.
        """
        predictions = self.model.predict(x)
        return predictions


def predict(coin_name: str, x: np.array) -> np.array:
    """
    Predict future prices from input data based on saved model.

    Parameters
    ----------
    coin_name : string
        Coin of interest.
    x : numpy array
        Input data.

    Returns
    -------
    predictions : numpy array
        Predicted future prices.
    """
    loaded_model = load_object(f'{modelSavedPath}_{coin_name}.sav')  # Load the saved model
    predictions = loaded_model.predict(x)  # Predict prices
    return predictions


def main(coin_name: str, fully_qualified_filepath: str) -> None:
    """
    Generate model

    Parameters
    ----------
    coin_name : string
        Coin of interest.
    fully_qualified_filepath : string
        Path to csv file with related data for coin.
    """
    logging.info(f"starting up {coin_name} sample sentiment model")

    dataframe = pd.read_csv(fully_qualified_filepath)  # Read in dataset

    model = SampleSentimentModel(coin_name, dataframe, 'open')  # Initiate model

    predictions = model.predict(model.x_test)  # Get predictions from test data

    # Generate classification report from actual and predicted prices
    logging.info(classification_report(model.y_test, predictions))

    # Log the actual and predicted prices
    logging.info("Predictions:")
    logging.info(f'{predictions}\n')
    logging.info("Actual Prices:")
    logging.info(f'{model.y_test}\n')


if __name__ == "__main__":
    """
    Loop over available coins to generate a saved model and metrics for each.
    """
    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)
