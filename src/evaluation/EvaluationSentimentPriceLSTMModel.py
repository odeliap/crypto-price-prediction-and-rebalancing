"""
Copy of SentimentPriceLSTMModel with slight modifications for evaluation against constant-sentiment
input datasets.
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

from src.models.Utils import save_model, save_scaler

# Set logging level
logging.basicConfig(level=logging.INFO)

# Set warnings level
warnings.filterwarnings('ignore')

# ------------- Constants -------------

train_test_split = 0.90  # Fractional split between training and testing

n_epochs = 1000  # Number of epochs for training
learning_rate = 0.001  # Learning rate for training

input_size = 8  # Number of input features
hidden_size = 2  # Number of features in hidden state
num_layers = 1  # Number of stacked lstm layers

# Paths for saving model and scalers
modelSavedPath = './outputs/models/SentimentPriceLSTMModel'
ssScalerSavedPath = './outputs/scalers/SentimentPriceLSTMSsScaler'
mmScalerSavedPath = './outputs/scalers/SentimentPriceLSTMMmScaler'

plot_style = 'seaborn-v0_8-pastel'  # Plot style

coins = {'bitcoin': [100, 50], 'ethereum': [50, 25], 'solana': [20, 10]}  # available coins

# ------------- Class -------------


class EvaluationSentimentPriceLSTM(nn.Module):

    def __init__(
        self,
        num_steps_out: int,
        in_size: int,
        hidden_layer_size: int,
        n_layers: int
    ) -> None:
        """
        Initialize EvaluationSentimentPriceLSTM

        Parameters
        ----------
        num_steps_out : int
            Number of output steps / size of the output.
        in_size : int
            Number of input steps / size of the input.
        hidden_layer_size : int
            Size of the hidden state.
        n_layers : int
            Number of stacked lstm layers.
        """
        super().__init__()
        self.num_classes = num_steps_out  # Set the number of classes to the size of the output
        self.num_layers = n_layers  # Set the number of recurrent layers in the lstm
        self.input_size = in_size  # Set the input size
        self.hidden_size = hidden_layer_size  # Set the number of neurons in each lstm layer

        # Initialize the lstm model
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_layer_size,
                            num_layers=n_layers, batch_first=True, dropout=0.2)
        # Initialize Linear to make the lstm fully connected
        # with hidden_size number of input features and
        # 128 output features
        self.fc_1 = nn.Linear(hidden_size, 128)
        # Initialize Linear to fully connect the last layer
        # with 128 input features and num_steps_out output features
        self.fc_2 = nn.Linear(128, num_steps_out)
        # Apply the rectified linear unit function element-wise
        # Replace all the negative elements in the input tensor with 0
        # and leave non-negative elements unchanged
        self.relu = nn.ReLU()

    def forward(self, x: np.array) -> torch.Tensor:
        """
        Performs lstm forward pass.

        Parameters
        ----------
        x : numpy array
            Input data.

        Returns
        -------
        Tensor
            Output tensor.
        """
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # Hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # Cell state

        # Propagate input through lstm
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # Get output, hidden, and cell state at next step
        hn = hn.view(-1, self.hidden_size)  # Reshape the hidden state at next step for dense layer
        out = self.relu(hn)   # Apply the rectified linear unit function element-wise
        out = self.fc_1(out)  # Get first dense layer
        out = self.relu(out)  # Apply the rectified linear unit function element-wise
        out = self.fc_2(out)  # Get the final output by fully connecting the last layer
        return out


def split_sequences(
    input_sequences: np.ndarray,
    output_sequence: np.ndarray,
    num_steps_in: int,
    num_steps_out: int
) -> (np.array, np.array):
    """
    Split multivariate sequence into past and future samples (X and y parts).

    Parameters
    ----------
    input_sequences : ndarray array
        Input sequences.
    output_sequence : ndarray array
        Output sequence.
    num_steps_in : int
        Number of input steps.
    num_steps_out : int
        Number of outputs steps.

    Returns
    -------
    X_arr
        Numpy array of x (input) data
    y_arr
        Numpy array of y (output) data
    """
    x, y = list(), list()  # Instantiate X and y lists
    # Loop over the length of the input sequences
    for i in range(len(input_sequences)):
        # Find the end of the input, output sequence
        end_index = i + num_steps_in
        output_end_index = end_index + num_steps_out - 1
        # Check if we are beyond the dataset
        if output_end_index > len(input_sequences):
            break
        # Gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_index], output_sequence[end_index-1:output_end_index, -1]
        x.append(seq_x), y.append(seq_y)
    x_arr = np.array(x)
    y_arr = np.array(y)
    return x_arr, y_arr


def training_loop(
    num_epochs: int,
    lstm: EvaluationSentimentPriceLSTM,
    optimiser: torch.optim.Adam,
    loss_fn: torch.nn.MSELoss,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """
    Train the lstm model.

    Parameters
    ----------
    num_epochs : int
        Number of epochs for training.
    lstm : EvaluationSentimentPriceLSTM object
        Lstm to model to train.
    optimiser : adam optimiser
        Optimiser for training.
    loss_fn : loss function
        Loss function.
    x_train : Tensor
        Input data for training.
    y_train : Tensor
        Output data for training.
    x_test : Tensor
        Testing input data.
    y_test : Tensor
        Testing output data.
    """
    for epoch in range(num_epochs):
        lstm.train()
        outputs = lstm.forward(x_train)  # Perform forward pass
        optimiser.zero_grad()  # Calculate the gradient, manually setting it to 0
        # Obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward()  # Calculate the loss of the loss function
        optimiser.step()  # Improve from the loss with backpropogation
        # Test the loss
        lstm.eval()
        test_preds = lstm(x_test)
        test_loss = loss_fn(test_preds, y_test)
        if epoch % 100 == 0:
            print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch,
                                                                      loss.item(),
                                                                      test_loss.item()))


def main(coin_name: str, fully_qualified_filepath: str, num_steps_in: int, num_steps_out: int) -> None:
    """
    Create lstm model, train it, and show evaluation of its performance

    Parameters
    ----------
    coin_name : string
        Coin of interest.
    fully_qualified_filepath : string
        Fully qualified filepath to dataset.
    num_steps_in : int
        Number of input steps for prediction.
    num_steps_out : int
        Number of output steps for prediction.
    """
    # Read in the dataset
    df = pd.read_csv(
        fully_qualified_filepath,
        header=0,
        low_memory=False,
        infer_datetime_format=True,
        index_col=['timestamp']
    )
    df = df.drop(df.columns[[0]], axis=1)

    # Get input and output data
    # where the output data is the open price
    x, y = df.drop(columns=['open']), df.open.values

    # Set the scalars
    mm = MinMaxScaler()
    ss = StandardScaler()

    # Get fitted x and y data
    x_trans = ss.fit_transform(x)
    y_trans = mm.fit_transform(y.reshape(-1, 1))

    # Get the input and output data split into sequences
    # of lengths corresponding to the number of input and output steps respectively
    x_ss, y_mm = split_sequences(x_trans, y_trans, num_steps_in, num_steps_out)

    # Get the total number of samples for training
    total_samples = len(x)
    # Set the train/test cutoff
    train_test_cutoff = round(train_test_split * total_samples)

    # Get the train and test data
    x_train = x_ss[:train_test_cutoff]
    x_test = x_ss[train_test_cutoff:]
    y_train = y_mm[:train_test_cutoff]
    y_test = y_mm[train_test_cutoff:]

    # Convert the training and testing data to pytorch tensors
    x_train_tensors = Variable(torch.Tensor(x_train))
    x_test_tensors = Variable(torch.Tensor(x_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # Reshape the training and testing to rows, timestamps, and features
    x_train_tensors_final = torch.reshape(
        x_train_tensors,
        (x_train_tensors.shape[0], num_steps_in, x_train_tensors.shape[2])
    )
    x_test_tensors_final = torch.reshape(
        x_test_tensors,
        (x_test_tensors.shape[0], num_steps_in, x_test_tensors.shape[2])
    )

    # Initiate the lstm model
    lstm = EvaluationSentimentPriceLSTM(
        num_steps_out,
        input_size,
        hidden_size,
        num_layers
    )

    # Define the training parameters
    loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # adam optimiser

    # Train the lstm model
    training_loop(num_epochs=n_epochs,
                  lstm=lstm,
                  optimiser=optimiser,
                  loss_fn=loss_fn,
                  x_train=x_train_tensors_final,
                  y_train=y_train_tensors,
                  x_test=x_test_tensors_final,
                  y_test=y_test_tensors)

    # Save the trained lstm model
    save_model(lstm, f'{modelSavedPath}_{coin_name}.sav')

    # Get the old transformers
    df_x_ss = ss.transform(df.drop(columns=['open']))  # old transformers
    df_y_mm = mm.transform(df.open.values.reshape(-1, 1))  # old transformers
    # Split the sequence
    df_x_ss, df_y_mm = split_sequences(df_x_ss, df_y_mm, num_steps_in, num_steps_out)
    # Convert to tensors
    df_x_ss = Variable(torch.Tensor(df_x_ss))
    df_y_mm = Variable(torch.Tensor(df_y_mm))
    # Reshape the dataset
    df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], num_steps_in, df_x_ss.shape[2]))

    # Get the predicted data
    train_predict = lstm(df_x_ss)  # forward pass
    data_predict = train_predict.data.numpy()  # numpy conversion
    datay_plot = df_y_mm.data.numpy()

    # Reverse the transformation and plot
    data_predict = mm.inverse_transform(data_predict)
    datay_plot = mm.inverse_transform(datay_plot)
    true, preds = [], []
    for i in range(len(datay_plot)):
        true.append(datay_plot[i][0])
    for i in range(len(data_predict)):
        preds.append(data_predict[i][0])

    # Plot the whole plot
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    plt.axvline(x=train_test_cutoff, c='r', linestyle='--')  # split graph into training and testing data with a line

    plt.plot(true, label='Actual Data')  # Add actual data
    plt.plot(preds, label='Predicted Data')  # Add predicted data
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f'{coin_name.capitalize()} Static-Sentiment Time-Series Prediction')
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_static_sentiment_whole_plot_{coin_name}.png", dpi=300)
    plt.show()

    test_predict = lstm(x_test_tensors_final[-1].unsqueeze(0))  # Get the last sample from the predicted data
    test_predict = test_predict.detach().numpy()
    test_predict = mm.inverse_transform(test_predict)
    test_predict = test_predict[0].tolist()

    test_target = y_test_tensors[-1].detach().numpy()  # Get the last sample from the test data
    test_target = mm.inverse_transform(test_target.reshape(1, -1))
    test_target = test_target[0].tolist()

    # Plot the small plot
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    plt.plot(test_target, label="Actual Data")
    plt.plot(test_predict, label="LSTM Predictions")
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f'{coin_name.capitalize()} Static-Sentiment LSTM Predictions vs Actual Data')
    plt.legend()
    plt.savefig(f"outputs/graphs/SentimentPriceLSTMModel_static_sentiment_small_plot_{coin_name}.png", dpi=300)
    plt.show()

    # Plot a one-shot multistep prediction
    plt.style.use(plot_style)
    plt.figure(figsize=(10, 6))
    y_plot_start = int(len(y) * 0.75)
    a = [x for x in range(y_plot_start, len(y))]
    plt.plot(a, y[y_plot_start:], label='Actual data')
    c = [x for x in range(len(y) - num_steps_out, len(y))]
    plt.plot(c, test_predict, label=f'One-Shot Multi-Step Prediction')
    plt.axvline(x=len(y) - num_steps_out, c='r', linestyle='--')
    plt.xlabel("Time[Days]")
    plt.ylabel("Price (USD)")
    plt.title(f"{coin_name.capitalize()} Static-Sentiment One-Shot Multi-Step Prediction ({num_steps_out} Days)")
    plt.legend()
    plt.savefig(
        f"outputs/graphs/SentimentPriceLSTMModel_static_sentiment_one_shot_multi_step_prediction_{coin_name}.png",
        dpi=300
    )
    plt.show()

    save_scaler(mm, f'{ssScalerSavedPath}_{coin_name}.pkl')
    save_scaler(ss, f'{mmScalerSavedPath}_{coin_name}.pkl')


if __name__ == "__main__":
    """
    Create a model and run the evaluation for each coin available.
    """
    for coin in coins.keys():
        filepath = f'datasets/{coin}_constant_sentiment_dataset.csv'
        n_steps_in = coins.get(coin)[0]
        n_steps_out = coins.get(coin)[1]
        main(coin, filepath, n_steps_in, n_steps_out)
