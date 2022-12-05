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

from src.models.Utils import save_model, save_scaler, load_model, load_scaler

# Set logging level
logging.basicConfig(level=logging.INFO)

# Set warnings level
warnings.filterwarnings('ignore')

# ------------- Constants -------------

n_epochs = 1000 # Number of epochs for training
learning_rate = 0.001 # Learning rate for training

input_size = 8 # Number of input features
hidden_size = 2 # Number of features in hidden state
num_layers = 1 # Number of stacked lstm layers

# Paths for saving model and scalers
modelSavedPath = f'outputs/models/SentimentPriceLSTMModel'
ssScalerSavedPath = f'outputs/scalers/SentimentPriceLSTMSsScaler'
mmScalerSavedPath = f'outputs/scalers/SentimentPriceLSTMMmScaler'


plot_style = 'seaborn-v0_8-pastel' # Plot style

n_steps_in = 30
n_steps_out = 15

coins = {'bitcoin': 0.95, 'ethereum': 0.90, 'solana': 0.70} # available coins to number of input/output steps

# ------------- Class -------------

class SentimentPriceLSTM(nn.Module):

    def __init__(
        self,
        n_steps_out: int,
        input_size: int,
        hidden_size: int,
        num_layers: int
    ) -> None:
        """
        Initialize EvaluationSentimentPriceLSTM

        Parameters
        ----------
        n_steps_out : int
            Number of output steps / size of the output.
        input_size : int
            Number of input steps / size of the input.
        hidden_size : int
            Size of the hidden state.
        num_layers : int
            Number of stacked lstm layers.
        """
        super().__init__()
        self.num_classes = n_steps_out  # Set the number of classes to the size of the output
        self.num_layers = num_layers  # Set the number of recurrent layers in the lstm
        self.input_size = input_size  # Set the input size
        self.hidden_size = hidden_size  # Set the number of neurons in each lstm layer

        # Initialize the lstm model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        # Initialize Linear to make the lstm fully connected
        # with hidden_size number of input features and
        # 128 output features
        self.fc_1 = nn.Linear(hidden_size, 128)
        # Initialize Linear to fully connect the last layer
        # with 128 input features and n_steps_out output features
        self.fc_2 = nn.Linear(128, n_steps_out)
        # Apply the rectified linear unit function element-wise
        # Replace all the negative elements in the input tensor with 0
        # and leave non-negative elements unchanged
        self.relu = nn.ReLU()

    def forward(self, X: np.array) -> torch.Tensor:
        """
        Performs lstm forward pass.

        Parameters
        ----------
        X : numpy array
            Input data.

        Returns
        -------
        Tensor
            Output tensor.
        """
        h_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size)) # Hidden state
        c_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size)) # Cell state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(X, (h_0, c_0))  # Get output, hidden, and cell state at next step
        hn = hn.view(-1, self.hidden_size)  # Reshape the hidden state at next step for dense layer
        out = self.relu(hn)   # Apply the rectified linear unit function element-wise
        out = self.fc_1(out)  # Get first dense layer
        out = self.relu(out)  # Apply the rectified linear unit function element-wise
        out = self.fc_2(out)  # Get the final output by fully connecting the last layer
        return out


def split_sequences(
    input_sequences: np.ndarray,
    output_sequence: np.ndarray,
    n_steps_in: int,
    n_steps_out: int
) -> (np.array, np.array):
    """
    Split multivariate sequence into past and future samples (X and y parts).

    Parameters
    ----------
    input_sequences : ndarray array
        Input sequences.
    output_sequence : ndarray array
        Output sequence.
    n_steps_in : int
        Number of input steps.
    n_steps_out : int
        Number of outputs steps.

    Returns
    -------
    X_arr
        Numpy array of x (input) data
    y_arr
        Numpy array of y (output) data
    """
    X, y = list(), list() # Instantiate X and y
    # Loop over the length of the input sequences
    for i in range(len(input_sequences)):
        # Find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # Check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # Gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def training_loop(
    n_epochs: int,
    lstm: SentimentPriceLSTM,
    optimiser: torch.optim.Adam,
    loss_fn: torch.nn.MSELoss,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """
    Train the lstm model.

    Parameters
    ----------
    n_epochs : int
        Number of epochs for training.
    lstm : SentimentPriceLSTM
        Lstm to model to train.
    optimiser : adam optimiser
        Optimiser for training.
    loss_fn : loss function
        Loss function.
    X_train : Tensor
        Input data for training.
    y_train : Tensor
        Output data for training.
    X_test : Tensor
        Testing input data.
    y_test : Tensor
        Testing output data.
    """
    for epoch in range(n_epochs):
        lstm.train()
        outputs = lstm.forward(X_train) # Perform forward pass
        optimiser.zero_grad() # Calculate the gradient, manually setting it to 0
        # Obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # Calculate the loss of the loss function
        optimiser.step() # Improve from the loss with backpropogation
        # Test the loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 100 == 0:
            logging.info("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch,
                                                                      loss.item(),
                                                                      test_loss.item()))

def predict(
    input: pd.DataFrame,
    coin: str,
    n_steps_in: int,
    model_saved_path: str,
    ss_scaler_saved_path: str,
    mm_scaler_saved_path: str
) -> np.ndarray:
    """
    Make predictions.

    Parameters
    ----------
    input : pandas dataframe
        Input data.
    coin : string
        Coin of interest.
    n_steps_in : int
        Number of input steps for prediction.
    model_saved_path : string
        Path to saved model.
    ss_scaler_saved_path : string
        Path to saved ss scaler.
    mm_scaler_saved_path : string
        Path to saved mm scaler.

    Returns
    -------
    predictions : ndarray
        Output predictions
    """
    print(f'Loading model from path {modelSavedPath}')
    lstm = load_model(f'{model_saved_path}_{coin}.sav') # load the lstm model
    ss = load_scaler(f'{ss_scaler_saved_path}_{coin}.pkl') # load the ss scaler
    mm = load_scaler(f'{mm_scaler_saved_path}_{coin}.pkl') # load the mm scaler

    X_trans = ss.fit_transform(input) # transformed input data
    X_tensors = Variable(torch.Tensor(X_trans))
    X_train_tensors_final = torch.reshape(X_tensors, (X_tensors.shape[0], n_steps_in, X_tensors.shape[2])) # reshape the transformed variable input data

    train_predict = lstm(X_train_tensors_final)  # perform forward pass
    predictions = train_predict.data.numpy()  # numpy conversion
    predictions = mm.inverse_transform(predictions)  # reverse transformation with mm scaler
    return predictions


def plot_price_over_time(coin: str, dataframe: pd.DataFrame) -> None:
    """
    Create and display plot of price over time.

    Parameters
    __________
    coin : string
        Coin of interest.
    dataframe : pandas dataframe
        Dataset to plot.
    """
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    plt.plot(dataframe.open)
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f"{coin.capitalize()} Price Over Time")
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_price_over_time_{coin}.png", dpi=300) # Save the figure to outputs directory
    plt.show()


def main(coin: str, filepath: str, n_steps_in: int, n_steps_out: int, train_test_split) -> None:
    """
    Create lstm model, train it, and show evaluation of its performance

    Parameters
    ----------
    coin : string
        Coin of interest.
    filepath : string
        Filepath to dataset.
    n_steps_in : int
        Number of input steps for prediction.
    n_steps_out : int
        Number of output steps for prediction.
    """
    # Read in the dataset
    df = pd.read_csv(filepath, header=0, low_memory=False, infer_datetime_format=True, index_col=['timestamp'])
    df = df.drop(df.columns[[0]], axis=1)

    # Plot the price over time
    plot_price_over_time(coin, df)

    # Get input and output data
    # where the output data is the open price
    X, y = df.drop(columns=['open']), df.open.values

    # Set the scalars
    mm = MinMaxScaler()
    ss = StandardScaler()

    # Get fitted x and y data
    X_trans = ss.fit_transform(X)
    y_trans = mm.fit_transform(y.reshape(-1, 1))

    # Get the input and output data split into sequences
    # of lengths corresponding to the number of input and output steps respectively
    X_ss, y_mm = split_sequences(X_trans, y_trans, n_steps_in, n_steps_out)

    # Get the total number of samples for training
    total_samples = len(X)
    # Set the train/test cutoff
    train_test_cutoff = round(train_test_split * total_samples)

    # Get the train and test data
    X_train = X_ss[:train_test_cutoff]
    X_test = X_ss[train_test_cutoff:]
    y_train = y_mm[:train_test_cutoff]
    y_test = y_mm[train_test_cutoff:]

    # Convert the training and testing data to pytorch tensors
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # Reshape the training and testing to rows, timestamps, and features
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], n_steps_in, X_train_tensors.shape[2]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], n_steps_in, X_test_tensors.shape[2]))

    # Initiate the lstm model
    lstm = SentimentPriceLSTM(n_steps_out,
                input_size,
                hidden_size,
                num_layers)

    # Define the training parameters
    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate) # adam optimiser

    # Train the model
    training_loop(n_epochs=n_epochs,
                  lstm=lstm,
                  optimiser=optimiser,
                  loss_fn=loss_fn,
                  X_train=X_train_tensors_final,
                  y_train=y_train_tensors,
                  X_test=X_test_tensors_final,
                  y_test=y_test_tensors)

    # Save the trained lstm model
    save_model(lstm, f'{modelSavedPath}_{coin}.sav')

    # Get the old transformers
    df_X_ss = ss.transform(df.drop(columns=['open']))  # old transformers
    df_y_mm = mm.transform(df.open.values.reshape(-1, 1))  # old transformers
    # Split the sequence
    df_X_ss, df_y_mm = split_sequences(df_X_ss, df_y_mm, n_steps_in, n_steps_out)
    # Convert to tensors
    df_X_ss = Variable(torch.Tensor(df_X_ss))
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    # Reshsape the dataset
    df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], n_steps_in, df_X_ss.shape[2]))

    # Get the predicted data
    train_predict = lstm(df_X_ss)  # forward pass
    data_predict = train_predict.data.numpy()  # numpy conversion
    dataY_plot = df_y_mm.data.numpy()

    # Reverse the transformation and plot
    data_predict = mm.inverse_transform(data_predict)
    dataY_plot = mm.inverse_transform(dataY_plot)
    true, preds = [], []
    for i in range(len(dataY_plot)):
        true.append(dataY_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot the whole plot
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=train_test_cutoff, c='r', linestyle='--')  # split graph into training and testing data with a line

    plt.plot(true, label='Actual Data')  # Add actual data
    plt.plot(preds, label='Predicted Data')  # Add predicted data
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f'{coin.capitalize()} Time-Series Prediction')
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_whole_plot_{coin}.png", dpi=300)
    plt.show()

    test_predict = lstm(X_test_tensors_final[-1].unsqueeze(0))  # Get the last sample from the predicted data
    test_predict = test_predict.detach().numpy()
    test_predict = mm.inverse_transform(test_predict)
    test_predict = test_predict[0].tolist()

    test_target = y_test_tensors[-1].detach().numpy()  # Get the last sample from the test data
    test_target = mm.inverse_transform(test_target.reshape(1, -1))
    test_target = test_target[0].tolist()

    # Plot small plot
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    plt.plot(test_target, label="Actual Data")
    plt.plot(test_predict, label="LSTM Predictions")
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f'{coin.capitalize()} LSTM Predictions vs Actual Data')
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_small_plot_{coin}.png", dpi=300)
    plt.show()

    # Plot a one-shot multi-step prediction
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    y_plot_start = int(len(y) * 0.75)
    a = [x for x in range(y_plot_start, len(y))]
    plt.plot(a, y[y_plot_start:], label='Actual data')
    c = [x for x in range(len(y) - n_steps_out, len(y))]
    plt.plot(c, test_predict, label=f'One-Shot Multi-Step Prediction')
    plt.axvline(x=len(y) - n_steps_out, c='r', linestyle='--')
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f"{coin.capitalize()} One-Shot Multi-Step Prediction ({n_steps_out} Days)")
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_one_shot_multi_step_prediction_{coin}.png", dpi=300)
    plt.show()

    save_scaler(mm, f'{ssScalerSavedPath}_{coin}.pkl')
    save_scaler(ss, f'{mmScalerSavedPath}_{coin}.pkl')


if __name__ == "__main__":
    """
    Create a model and display metrics for each coin available.
    """
    for coin in coins.keys():
        filepath = f'../sentiment_analysis/outputs/{coin}_sentiment_dataset.csv'
        train_test_split = coins[coin]
        main(coin, filepath, n_steps_in, n_steps_out, train_test_split)