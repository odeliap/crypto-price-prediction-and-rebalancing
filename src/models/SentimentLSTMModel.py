"""
LSTM neural network with date and sentiment inputs.

Made by following tutorial:
https://stackoverflow.com/questions/59457567/how-can-i-create-multiple-input-one-output-lstm-model-with-keras
"""

# ------------- Libraries -------------
import logging

import pandas as pd

import datetime

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import Adam
# for bletchley change to:
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from Utils import saveModel, comparisonGraph

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

train_split = 0.90

modelSavedPath = './outputs/SentimentLSTMModel'

# ------------- Class -------------

class SentimentLSTMModel:

    def __init__(self, dataframe: pd.DataFrame, coin: str):
        """
        Initialize SentimentLSTMModel

        :param dataframe: pandas dataframe to process
        :type: pd.DataFrame

        :param coin: coin of interest
        :type: str
        """
        self.dataframe = dataframe

        self.scaler = MinMaxScaler(feature_range=(0,1))

        self.coin = coin

        train_size = int(len(dataframe) * train_split)
        test_size = len(dataframe) - train_size

        new_data = self.dataframe.loc[:,
                   ['open', 'timestamp', 'subjectivity', 'polarity', 'compound', 'negative', 'neutral', 'positive']]
        new_data.info()

        date = new_data.timestamp.values
        dates = []
        for i in date:
            dates.append(i.split('-')[0])
        new_data['timestamp'] = dates

        timestamp_train, timestamp_test, subjectivity_train, subjectivity_test, polarity_train, polarity_test, compound_train, compound_test, negative_train, negative_test, neutral_train, neutral_test, positive_train, positive_test, open_train, open_test = self.preprocess(new_data, train_size, test_size)

        logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        self.rnn = self.buildModel(1)
        self.rnn.fit([timestamp_train, subjectivity_train, polarity_train, compound_train, negative_train, neutral_train, positive_train],
                [open_train],
                validation_data = ([timestamp_test, subjectivity_test, polarity_test, compound_test, negative_test, neutral_test, positive_test], [open_test]),
                epochs = 1,
                batch_size = 10,
                callbacks = [tensorboard_callback]
                )
        saveModel(self.rnn, f'{modelSavedPath}_{coin}.sav')

        scaled_result = self.rnn.predict([timestamp_test, subjectivity_test, polarity_test, compound_test, negative_test, neutral_test, positive_test])
        result = self.scaler.inverse_transform(scaled_result)

        self.report_and_graph_test_results(coin, result, open_test)


    def report_and_graph_test_results(self, coin, result, test_output):
        """
        Report and graph test results.

        :param coin: coin of interest
        :type: str

        :param result: 2d predicted outputs array
        :type: 2d array-like

        :param test_output: 2d test outputs (actual prices set aside for testing)
        :type: 2d array-like
        """
        predictions = []
        test_actual_prices_2d = []
        test_actual_prices_1d = []

        for arr in result:
            for val in arr:
                predictions.append(val)

        for arr in test_output:
            for sub_arr in arr:
                test_actual_prices_2d.append(sub_arr)

        test_actual_prices_2d = self.scaler.inverse_transform(test_actual_prices_2d)

        for arr in test_actual_prices_2d:
            for val in arr:
                test_actual_prices_1d.append(val)

        logging.info("PREDICTIONS:")
        logging.info(f'{predictions}\n')
        logging.info("ACTUAL PRICES:")
        logging.info(f'{test_actual_prices_1d}\n')

        comparisonGraph(test_actual_prices_1d, predictions, coin,
                        f'outputs/graphs/SentimentLSTMModel_comparison_{coin}.png')


    def preprocess(self, dataframe, train_size, test_size):
        """
        Preprocess data.

        :param dataframe: dataframe to process

        :param train_size: size of training set

        :param test_size: size of testing set

        :return timestamp_train, timestamp_test, subjectivity_train, subjectivity_test, polarity_train, polarity_test, compound_train, compound_test, negative_train, negative_test, neutral_train, neutral_test, positive_train, positive_test, open_train, open_test: training and testing sets for each column
        """
        timestamp = dataframe['timestamp'].values.reshape(-1, 1)
        subjectivity = dataframe['subjectivity'].values.reshape(-1, 1)
        polarity = dataframe['polarity'].values.reshape(-1, 1)
        compound = dataframe['compound'].values.reshape(-1, 1)
        negative = dataframe['negative'].values.reshape(-1, 1)
        neutral = dataframe['neutral'].values.reshape(-1, 1)
        positive = dataframe['positive'].values.reshape(-1, 1)
        open = dataframe['open'].values.reshape(-1, 1)

        timestamp_ = self.scaler.fit_transform(timestamp)
        subjectivity_ = self.scaler.fit_transform(subjectivity)
        polarity_ = self.scaler.fit_transform(polarity)
        compound_ = self.scaler.fit_transform(compound)
        negative_ = self.scaler.fit_transform(negative)
        neutral_ = self.scaler.fit_transform(neutral)
        positive_ = self.scaler.fit_transform(positive)
        open_ = self.scaler.fit_transform(open)

        timestamp_train, timestamp_test = self.split_train_test(timestamp_, train_size, test_size)
        subjectivity_train, subjectivity_test = self.split_train_test(subjectivity_, train_size, test_size)
        polarity_train, polarity_test = self.split_train_test(polarity_, train_size, test_size)
        compound_train, compound_test = self.split_train_test(compound_, train_size, test_size)
        negative_train, negative_test = self.split_train_test(negative_, train_size, test_size)
        neutral_train, neutral_test = self.split_train_test(neutral_, train_size, test_size)
        positive_train, positive_test = self.split_train_test(positive_, train_size, test_size)
        open_train, open_test = self.split_train_test(open_, train_size, test_size)

        return timestamp_train, timestamp_test, subjectivity_train, subjectivity_test, polarity_train, polarity_test, compound_train, compound_test, negative_train, negative_test, neutral_train, neutral_test, positive_train, positive_test, open_train, open_test


    def split_train_test(self, obj, train_size, test_size):
        """
        Split object into training and testing data.

        :param obj: object to split

        :param train_size: size of training data
        :param test_size: size of testing data

        :return obj_train, obj_test
        """
        obj_train = obj[0:train_size].reshape(train_size, 1, 1)
        obj_test = obj[train_size:len(obj)].reshape(test_size, 1, 1)
        return obj_train, obj_test


    def buildModel(self, labelLength):
        """
        Build model.
        """
        timestamp = tf.keras.Input(shape=(1,1), name='timestamp')
        subjectivity = tf.keras.Input(shape=(1,1), name='subjectivity')
        polarity = tf.keras.Input(shape=(1,1), name='polarity')
        compound = tf.keras.Input(shape=(1, 1), name='compound')
        negative = tf.keras.Input(shape=(1, 1), name='negative')
        neutral = tf.keras.Input(shape=(1, 1), name='neutral')
        positive = tf.keras.Input(shape=(1, 1), name='positive')

        timestampLayers = LSTM(100, return_sequences=False)(timestamp)
        subjectivityLayers = LSTM(100, return_sequences=False)(subjectivity)
        polarityLayers = LSTM(100, return_sequences=False)(polarity)
        compoundLayers = LSTM(100, return_sequences=False)(compound)
        negativeLayers = LSTM(100, return_sequences=False)(negative)
        neutralLayers = LSTM(100, return_sequences=False)(neutral)
        positiveLayers = LSTM(100, return_sequences=False)(positive)

        output = tf.keras.layers.concatenate(inputs=[timestampLayers, subjectivityLayers, polarityLayers, compoundLayers, negativeLayers, neutralLayers, positiveLayers], axis=1)
        output = Dense(labelLength, activation='relu', name='weightedAverage_output_3')(output)

        model = Model(inputs=[timestamp, subjectivity, polarity, compound, negative, neutral, positive], outputs=[output])
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model


    def predict(self, input):
        """
        Make predictions.

        :param input: input or x data

        :return predictions: output predictions
        """
        return self.rnn.predict(input)


def main(filepath: str, coin: str):
    dataframe = pd.read_csv(filepath)
    columns = dataframe.columns
    drop_columns = []
    check_columns = ['close', 'high', 'low']
    for col in check_columns:
        if col in columns:
            drop_columns.append(col)
    dataframe = dataframe.drop(columns=drop_columns)
    logging.info(dataframe.head())

    SentimentLSTMModel(dataframe, coin)


if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)
