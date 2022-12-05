"""
LSTM neural network with date and sentiment inputs.

Made by following tutorial:
https://stackoverflow.com/questions/59457567/how-can-i-create-multiple-input-one-output-lstm-model-with-keras
"""

# ------------- Libraries -------------
import logging

import pandas as pd
import numpy as np

import datetime

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from Utils import save_model, save_scaler, generate_comparison_graph
from Unpickler import load_object

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

test_split = 0.90  # train test split

# Paths for saving model and scaler
modelSavedPath = './outputs/models/SentimentLSTMModel'
scalerSavedPath = './outputs/scalers/SentimentLSTMScaler'

coins = ['bitcoin', 'ethereum', 'solana']  # available coins

# ------------- Class -------------


class SentimentLSTMModel:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        coin_name: str
    ) -> None:
        """
        Initialize SentimentLSTMModel

        Parameters
        ----------
        dataframe : pandas dataframe
            Dataset to process.
        coin_name : string
            Coin of interest.
        """
        self.dataframe = dataframe
        self.coin = coin_name

        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Set scaler to min max scaler

        train_size = int(len(dataframe) * test_split)  # Set training data size
        test_size = len(dataframe) - train_size  # Set testing data size

        # Get all the input features and the output price columns
        new_data = self.dataframe.loc[
                   :, ['open', 'timestamp', 'subjectivity', 'polarity', 'compound', 'negative', 'neutral', 'positive']]

        # Clean timestamp columns
        date = new_data.timestamp.values
        dates = []
        for i in date:
            dates.append(i.split('-')[0])
        new_data['timestamp'] = dates

        # Get training and testing data for each input feature and the output prices
        timestamp_train, timestamp_test, subjectivity_train, subjectivity_test, polarity_train, polarity_test, \
            compound_train, compound_test, negative_train, negative_test, neutral_train, neutral_test, positive_train, \
            positive_test, open_train, open_test = self.preprocess(new_data, train_size, test_size)

        # Initialize logs
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        # Initialize recurrent neural network
        self.rnn = self.build_model(1)
        # Fit the input features and output prices to the model
        self.rnn.fit(
            [timestamp_train, subjectivity_train, polarity_train, compound_train, negative_train, neutral_train,
             positive_train],
            [open_train],
            validation_data=([timestamp_test, subjectivity_test, polarity_test, compound_test, negative_test,
                              neutral_test, positive_test], [open_test]),
            epochs=1,
            batch_size=10,
            callbacks=[tensorboard_callback]
        )

        # Make scaled predictions using the test data
        scaled_result = self.rnn.predict([timestamp_test, subjectivity_test, polarity_test, compound_test,
                                          negative_test, neutral_test, positive_test])
        # Transform the scaled predictions to actual prices
        result = self.scaler.inverse_transform(scaled_result)

        # Generate report and graphs for test results
        self.report_and_graph_test_results(coin_name, result, open_test)

        # Save the model and scaler
        save_model(self.rnn, f'{modelSavedPath}_{coin_name}.sav')
        save_scaler(self.scaler, f'{scalerSavedPath}_{coin_name}.pkl')

    def report_and_graph_test_results(
        self,
        coin_name: str,
        result: np.ndarray,
        test_output: np.ndarray
    ) -> None:
        """
        Report and graph test results.

        Parameters
        ----------
        coin_name : string
            Coin of interest.
        result : 2d array-like
            Predicted outputs array.
        test_output : 2d array-like
            Test outputs.
        """
        # Initialize lists to hold predictions and actual prices
        predictions = []
        test_actual_prices_2d = []
        test_actual_prices_1d = []

        # Add predicted values to predictions list
        for arr in result:
            for val in arr:
                predictions.append(val)

        # Add scaled real prices to 2d test list
        for arr in test_output:
            for sub_arr in arr:
                test_actual_prices_2d.append(sub_arr)

        # Transform scaled real prices to actual real prices
        test_actual_prices_2d = self.scaler.inverse_transform(test_actual_prices_2d)

        # Add actual real prices to 1d test list
        for arr in test_actual_prices_2d:
            for val in arr:
                test_actual_prices_1d.append(val)

        # Log results
        logging.info("Predictions:")
        logging.info(f'{predictions}\n')
        logging.info("Actual Prices:")
        logging.info(f'{test_actual_prices_1d}\n')

        # Graph actual prices against predicted prices
        generate_comparison_graph(test_actual_prices_1d, predictions, coin_name,
                                  f'outputs/graphs/SentimentLSTMModel_comparison_{coin_name}.png')

    def preprocess(
        self,
        dataframe: pd.DataFrame,
        train_size: int,
        test_size: int
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Preprocess data.

        Parameters
        ----------
        dataframe : pandas dataframe
            Dataframe to process.
        train_size : int
            Size of the training set.
        test_size : int
            Size of the testing set.

        Returns
        -------
        timestamp_train : 2d array-like
            Timestamp feature training set.
        timestamp_test : 2d array-like
            Timestamp feature testing set.
        subjectivity_train : 2d array-like
            Subjectivity feature training set.
        subjectivity_test : 2d array-like
            Subjectivity feature testing set.
        polarity_train : 2d array-like
            Polarity feature training set.
        polarity_test : 2d array-like
            Polarity feature testing set.
        compound_train : 2d array-like
            Compound feature training set.
        compound_test : 2d array-like
            Compound feature testing set.
        negative_train : 2d array-like
            Negative feature training set.
        negative_test : 2d array-like
            Negative feature testing set.
        neutral_train : 2d array-like
            Neutral feature training set.
        neutral_test : 2d array-like
            Neutral feature testing set.
        positive_train : 2d array-like
            Positive feature training set.
        positive_test : 2d array-like
            Positive feature testing set.
        open_train : 2d array-like
            Open output prices training set.
        open_test : Open output prices testing set.
        """
        # Get feature columns
        timestamp = dataframe['timestamp'].values.reshape(-1, 1)
        subjectivity = dataframe['subjectivity'].values.reshape(-1, 1)
        polarity = dataframe['polarity'].values.reshape(-1, 1)
        compound = dataframe['compound'].values.reshape(-1, 1)
        negative = dataframe['negative'].values.reshape(-1, 1)
        neutral = dataframe['neutral'].values.reshape(-1, 1)
        positive = dataframe['positive'].values.reshape(-1, 1)
        # Get output price column
        open_prices = dataframe['open'].values.reshape(-1, 1)

        # Transform features with scaler
        timestamp_ = self.scaler.fit_transform(timestamp)
        subjectivity_ = self.scaler.fit_transform(subjectivity)
        polarity_ = self.scaler.fit_transform(polarity)
        compound_ = self.scaler.fit_transform(compound)
        negative_ = self.scaler.fit_transform(negative)
        neutral_ = self.scaler.fit_transform(neutral)
        positive_ = self.scaler.fit_transform(positive)
        # Transform output prices with scaler
        open_prices_ = self.scaler.fit_transform(open_prices)

        # Split each feature into training ana testing set
        timestamp_train, timestamp_test = self.split_train_test(timestamp_, train_size, test_size)
        subjectivity_train, subjectivity_test = self.split_train_test(subjectivity_, train_size, test_size)
        polarity_train, polarity_test = self.split_train_test(polarity_, train_size, test_size)
        compound_train, compound_test = self.split_train_test(compound_, train_size, test_size)
        negative_train, negative_test = self.split_train_test(negative_, train_size, test_size)
        neutral_train, neutral_test = self.split_train_test(neutral_, train_size, test_size)
        positive_train, positive_test = self.split_train_test(positive_, train_size, test_size)
        # Split output prices into training and testing set
        open_train, open_test = self.split_train_test(open_prices_, train_size, test_size)

        return timestamp_train, timestamp_test, subjectivity_train, subjectivity_test, polarity_train, polarity_test, \
            compound_train, compound_test, negative_train, negative_test, neutral_train, neutral_test, \
            positive_train, positive_test, open_train, open_test

    @staticmethod
    def split_train_test(arr: np.array, train_size: int, test_size: int) -> np.ndarray:
        """
        Split object into training and testing data.

        Parameters
        ----------
        arr : numpy array
            Array to split.
        train_size : int
            Size of the training set.
        test_size : int
            Size of the testing set.

        Returns
        -------
        arr_train : 2d array-like
            Training set.
        arr_test : 2d array-like
            Testing set.
        """
        arr_train = arr[0:train_size].reshape(train_size, 1, 1)
        arr_test = arr[train_size:len(arr)].reshape(test_size, 1, 1)

        return arr_train, arr_test

    @staticmethod
    def build_model(label_len: int) -> Model:
        """
        Build model.

        Parameters
        ----------
        label_len : int
            Length of labels.

        Returns
        -------
        model : keras model
            Rnn model.
        """
        # Create Input for each feature
        timestamp = tf.keras.Input(shape=(1, 1), name='timestamp')
        subjectivity = tf.keras.Input(shape=(1, 1), name='subjectivity')
        polarity = tf.keras.Input(shape=(1, 1), name='polarity')
        compound = tf.keras.Input(shape=(1, 1), name='compound')
        negative = tf.keras.Input(shape=(1, 1), name='negative')
        neutral = tf.keras.Input(shape=(1, 1), name='neutral')
        positive = tf.keras.Input(shape=(1, 1), name='positive')

        # Create layer for each feature
        timestamp_layers = LSTM(100, return_sequences=False)(timestamp)
        subjectivity_layers = LSTM(100, return_sequences=False)(subjectivity)
        polarity_layers = LSTM(100, return_sequences=False)(polarity)
        compound_layers = LSTM(100, return_sequences=False)(compound)
        negative_layers = LSTM(100, return_sequences=False)(negative)
        neutral_layers = LSTM(100, return_sequences=False)(neutral)
        positive_layers = LSTM(100, return_sequences=False)(positive)

        # Concatenate feature layers
        output = tf.keras.layers.concatenate(
            inputs=[timestamp_layers, subjectivity_layers, polarity_layers, compound_layers, negative_layers,
                    neutral_layers, positive_layers],
            axis=1
        )
        output = Dense(label_len, activation='relu', name='weightedAverage_output_7')(output)

        # Generate model with concatenated feature inputs
        model = Model(
            inputs=[timestamp, subjectivity, polarity, compound, negative, neutral, positive],
            outputs=[output]
        )

        # User adam optimiser
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

        # Compile model with adam optimiser and mean-squared-error loss
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def predict(self, x: np.array) -> np.array:
        """
        Predict future prices for input data.

        Parameters
        ----------
        x : numpy array
            Input data.

        Returns
        -------
        predictions : numpy array
            Predicted future prices.
        """
        scaled_predictions = self.rnn.predict(x)  # Get scaled predicted prices
        predictions = self.scaler.inverse_transform(scaled_predictions)  # Convert scaled prices to actual prices
        return predictions


def predict(x: np.array, coin_name: str) -> np.array:
    """
    Predict future prices for input data on saved model.

    Parameters
    ----------
    x : numpy array
        Input data.
    coin_name : string
        Coin of interest.

    Returns
    -------
    predictions : numpy array
        Predicted future prices.
    """
    # Load saved model and scaler
    model = load_object(f'{modelSavedPath}_{coin_name}.sav')
    scaler = load_object(f'{scalerSavedPath}_{coin_name}.pkl')

    scaled_predictions = model.predict(x)  # Get scaled predicted prices
    predictions = scaler.inverse_transform(scaled_predictions)  # Convert scaled prices to actual prices

    return predictions


def main(coin_name: str, fully_qualified_filepath: str) -> None:
    """
    Create SentimentLSTMModel for coin and dataset from filepath.

    Parameters
    __________
    coin_name : string
        Coin of interest.
    fully_qualified_filepath : string
        Filepath to dataset.
    """
    dataframe = pd.read_csv(fully_qualified_filepath)  # load csv from filepath

    columns = dataframe.columns  # Get dataframe columns

    drop_columns = []
    check_columns = ['close', 'high', 'low']

    # Add any of the columns in the check columns
    # that exist in the dataframe
    # to the drop columns
    for col in check_columns:
        if col in columns:
            drop_columns.append(col)

    dataframe = dataframe.drop(columns=drop_columns)  # drop columns

    SentimentLSTMModel(dataframe, coin_name)  # Generate model for cleaned dataframe and coin


if __name__ == "__main__":
    """
    Loop over available coins to generate a saved model and metrics for each.
    """
    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)
