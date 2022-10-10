"""
LSTM neural network with date and sentiment inputs.

Made by following tutorial:
https://www.analyticsvidhya.com/blog/2020/10/multivariate-multi-step-time-series-forecasting-using-stacked-lstm-sequence-to-sequence-autoencoder-in-tensorflow-2-0-keras/
"""

# ------------- Libraries -------------
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import os

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

train_split = 0.80
look_back = 3

n_past = 10
n_future = 5
n_features = 7

# ------------- Class -------------

class SentimentLSTMModel:

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize SentimentLSTMModel

        :param dataframe: pandas dataframe to process
        :type: pd.DataFrame
        """
        self.dataframe = dataframe
        train, test = self.preprocess()

        self.X_train, self.y_train = self.split_input_output(train)
        self.X_test, self.y_test = self.split_input_output(test)


    def construct_e1d1_model(self):
        """
        Construct E1D1 model (Sequence to Sequence Model with one encoder layer and one decoder layer).
        """
        encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)

        encoder_states1 = encoder_outputs1[1:]

        decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

        decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state = encoder_states1)
        decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

        self.model_e1d1 = tf.keras.models.Model(encoder_inputs, decoder_outputs1)

        self.model_e1d1.summary()


    def construct_e2d2_model(self):
        """
        Construct E2D2 model (Sequence to Sequence Model with two encoder layers and two decoder layers).
        """
        encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
        encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
        encoder_outputs1 = encoder_l1(encoder_inputs)
        encoder_states1 = encoder_outputs1[1:]
        encoder_l2 = tf.keras.layers.LSTM(100, return_state = True)
        encoder_outputs2 = encoder_l2(encoder_outputs1[0])
        encoder_states2 = encoder_outputs2[1:]

        decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])

        decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs, initial_state = encoder_states1)
        decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1, initial_state = encoder_states2)
        decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)

        self.model_e2d2 = tf.keras.models.Model(encoder_inputs, decoder_outputs2)
        self.model_e2d2.summary()


    def split_input_output(self, dataframe):
        """
        Split dataframe into input and output datasets.

        :param dataframe: dataframe to split
        :type: pd.DataFrame

        :return X, y: input and output datasets
        """
        X, y = self.split_series(dataframe.values, n_past, n_future)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        y = y.reshape((y.shape[0], y.shape[1], n_features))
        return X, y


    def split_data(self):
        """
        Split data into train and test dataframes.
        """
        dataframe_len = len(self.dataframe)
        train_len = int(dataframe_len * train_split)
        train_dataframe, test_dataframe = self.dataframe[1:train_len], self.dataframe[train_len:]
        return train_dataframe, test_dataframe


    def preprocess(self):
        """
        Preprocess data.

        :return train, test
        """
        scaler = MinMaxScaler(feature_range=(-1, 1))

        self.train_dataframe, self.test_dataframe = self.split_data()

        train = self.train_dataframe

        self.scalers = {}
        for i in train.columns:
            s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            self.scalers['scaler_' + i] = scaler
            train[i] = s_s

        test = self.test_dataframe

        for i in self.train_dataframe.columns:
            scaler = self.scalers['scaler_' + i]
            s_s = scaler.transform(test[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            self.scalers['scaler_' + i] = scaler
            test[i] = s_s

        return train, test


    @staticmethod
    def split_series(series, n_past, n_future):
        """
        Transform series into samples using sliding window approach.

        :param series: series to transform

        :param n_past: number of past observations

        :param n_future: number of future observations
        """
        X, y = list(), list()

        for window_start in range(len(series)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            past, future = series[window_start:past_end, :], series[past_end:future_end, :]
            X.append(past)
            y.append(future)
        return np.array(X), np.array(y)


    def train(self):
        """
        Train models.
        """
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
        self.model_e1d1.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        history_e1d1 = self.model_e1d1.fit(self.X_train, self.y_train, epochs=25, validation_data=(self.X_test, self.y_test), batch_size=32, verbose=0, callbacks=[reduce_lr])
        self.model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        history_e2d2 = self.model_e2d2.fit(self.X_train, self.y_train, epochs=25, validation_data=(self.X_test, self.y_test), batch_size=32, verbose=0, callbacks=[reduce_lr])


def main(filepath):
    dataframe = pd.read_csv(filepath, header=0, low_memory=False, infer_datetime_format=True, index_col=['timestamp'])
    columns = dataframe.columns
    drop_columns = []
    check_columns = ['close', 'high', 'low']
    for col in check_columns:
        if col in columns:
            drop_columns.append(col)
    dataframe = dataframe.drop(drop_columns)
    logging.info(dataframe.head())

    model_class = SentimentLSTMModel(dataframe)
    model_e1d1 = model_class.model_e1d1
    model_e2d2 = model_class.model_e2d2

    pred_e1d1 = model_e1d1.predict(model_class.X_test)
    pred_e2d2 = model_e2d2.predict(model_class.X_test)

    for index, i in enumerate(model_class.train_dataframe):
        scaler = model_class.scalers['scaler_' + i]
        pred_e1d1[:,:,index] = scaler.inverse_transform(pred_e1d1[:, :, index])
        pred_e2d2[:,:,index] = scaler.inverse_transform(pred_e2d2[:,:,index])
        model_class.y_train[:,:,index] = scaler.inverse_transform(model_class.y_train[:,:,index])
        model_class.y_test[:,:,index] = scaler.inverse_transform(model_class.y_test[:,:,index])

    for index, i in enumerate(model_class.train_dataframe.columns):
        logging.info(i)
        for j in range(1,6):
            logging.info(f'Day {j}:')
            logging.info(f'MAE-E1D1 : {mean_absolute_error(model_class.y_test[:,j-1,index],pred_e1d1[:,j-1,index])},')
            logging.info(f'MAE-E2D2 : {mean_absolute_error(model_class.y_test[:,j-1,index],pred_e2d2[:,j-1,index])}')
        logging.info("")
        logging.info("")




