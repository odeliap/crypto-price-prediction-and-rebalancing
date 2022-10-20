"""
LSTM neural network with date, historical prices, and sentiment input.

Made by following tutorial:
https://charlieoneill.medium.com/predicting-the-price-of-bitcoin-with-multivariate-pytorch-lstms-695bc294130
"""

# ------------- Libraries -------------
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.autograd import Variable
from torch import nn

from Utils import saveModel, saveScaler, loadModel, loadScaler

# Set logging level
logging.basicConfig(level=logging.INFO)

# Set warnings level
warnings.filterwarnings('ignore')

# ------------- Constants -------------

n_steps_in = 100
n_steps_out = 50
train_test_split = 0.90

n_epochs = 1000
learning_rate = 0.001

input_size = 8 # number of features
hidden_size = 2 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers

num_classes = n_steps_out # number of output classes

modelSavedPath = './outputs/models/SentimentPriceLSTMModel'
ssScalerSavedPath = './outputs/scalers/SentimentPriceLSTMSsScaler'
mmScalerSavedPath = './outputs/scalers/SentimentPriceLSTMMmScaler'

# ------------- Class -------------

class SentimentPriceLSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Lstm forward pass.
        """
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out


def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    """
    Split multivariate sequence into past and future samples (X and y).
    """
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,
                  X_test, y_test):
    """
    Train model.
    """
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 100 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch,
                                                                      loss.item(),
                                                                      test_loss.item()))

def predict(input, coin):
    """
    Make predictions.

    :param input: input or x data

    :param coin: coin of interest

    :return predictions: output predictions
    """
    lstm = loadModel(f'{modelSavedPath}_{coin}.sav')
    ss = loadScaler(f'{ssScalerSavedPath}_{coin}.pkl')
    mm = loadScaler(f'{mmScalerSavedPath}_{coin}.pkl')

    X_trans = ss.fit_transform(input)

    X_tensors = Variable(torch.Tensor(X_trans))

    X_train_tensors_final = torch.reshape(X_tensors, (X_tensors.shape[0], n_steps_in, X_tensors.shape[2]))

    train_predict = lstm(X_train_tensors_final)  # forward pass
    predictions = train_predict.data.numpy()  # numpy conversion
    predictions = mm.inverse_transform(predictions)  # reverse transformation
    return predictions

def main(coin: str, filepath: str):
    df = pd.read_csv(filepath, header=0, low_memory=False, infer_datetime_format=True, index_col=['timestamp'])
    df = df.drop(df.columns[[0]], axis=1)
    print(df.head())

    plt.plot(df.open)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title(f"{coin.capitalize()} price over time")
    plt.show()

    X, y = df.drop(columns=['open']), df.open.values
    print(X.shape, y.shape)

    mm = MinMaxScaler()
    ss = StandardScaler()

    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1))

    X_ss, y_mm = split_sequences(X_trans, y_trans, n_steps_in, n_steps_out)
    print(X_ss.shape, y_mm.shape)

    total_samples = len(X)
    train_test_cutoff = round(train_test_split * total_samples)

    X_train = X_ss[:train_test_cutoff]
    X_test = X_ss[train_test_cutoff:]

    y_train = y_mm[:train_test_cutoff]
    y_test = y_mm[train_test_cutoff:]

    print("Training shape:", X_train.shape, y_train.shape)
    print("Testing shape:", X_test.shape, y_test.shape)

    # convert to pytorch tensors
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # reshaping to rows, timestamps, features
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], n_steps_in, X_train_tensors.shape[2]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], n_steps_in, X_test_tensors.shape[2]))

    print("Training Shape:", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors.shape)

    X_check, y_check = split_sequences(X, y.reshape(-1, 1), n_steps_in, n_steps_out)
    print(X_check[-1][0:4])
    print(X.iloc[-149-145])
    print(y_check[-1])
    print(df.open.values[-n_steps_out:])

    lstm = SentimentPriceLSTM(num_classes,
                input_size,
                hidden_size,
                num_layers)

    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    training_loop(n_epochs=n_epochs,
                  lstm=lstm,
                  optimiser=optimiser,
                  loss_fn=loss_fn,
                  X_train=X_train_tensors_final,
                  y_train=y_train_tensors,
                  X_test=X_test_tensors_final,
                  y_test=y_test_tensors)

    saveModel(lstm, f'{modelSavedPath}_{coin}.sav')

    df_X_ss = ss.transform(df.drop(columns=['open']))  # old transformers
    df_y_mm = mm.transform(df.open.values.reshape(-1, 1))  # old transformers
    # split the sequence
    df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, n_steps_in, n_steps_out)
    # converting to tensors
    df_X_ss = Variable(torch.Tensor(df_X_ss))
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    # reshaping the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], n_steps_in, df_X_ss.shape[2]))

    train_predict = lstm(df_X_ss)  # forward pass
    data_predict = train_predict.data.numpy()  # numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    data_predict = mm.inverse_transform(data_predict)  # reverse transformation
    dataY_plot = mm.inverse_transform(dataY_plot)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])
    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=train_test_cutoff, c='r', linestyle='--')  # size of the training set

    plt.plot(true, label='Actual Data')  # actual plot
    plt.plot(preds, label='Predicted Data')  # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_whole_plot_{coin}.png", dpi=300)
    plt.show()

    test_predict = lstm(X_test_tensors_final[-1].unsqueeze(0))  # get the last sample
    test_predict = test_predict.detach().numpy()
    test_predict = mm.inverse_transform(test_predict)
    test_predict = test_predict[0].tolist()

    test_target = y_test_tensors[-1].detach().numpy()  # last sample again
    test_target = mm.inverse_transform(test_target.reshape(1, -1))
    test_target = test_target[0].tolist()

    plt.plot(test_target, label="Actual Data")
    plt.plot(test_predict, label="LSTM Predictions")
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_small_plot_{coin}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))  # plotting
    a = [x for x in range(2500, len(y))]
    plt.plot(a, y[2500:], label='Actual data')
    c = [x for x in range(len(y) - n_steps_out, len(y))]
    plt.plot(c, test_predict, label=f'One-shot multi-step prediction ({n_steps_out} days)')
    plt.axvline(x=len(y) - n_steps_out, c='r', linestyle='--')
    plt.legend()
    plt.show()

    saveScaler(mm, f'{ssScalerSavedPath}_{coin}.pkl')
    saveScaler(ss, f'{mmScalerSavedPath}_{coin}.pkl')


if __name__ == "__main__":
    coins = ['bitcoin', 'ethereum', 'solana']

    for coin in coins:
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        main(coin, filepath)