.. _pricelstmmodel:

Price LSTM Model
================

.. code-block:: python
    :name: class

    class PriceLSTMModel(coin: str, dataframe: pd.DataFrame, price_label: str)

Bases: :code:`object`

    .. code-block:: python
        :name: init

        _init_(coin: str, dataframe: pd.DataFrame, price_label: str) -> None

    PARAMETERS:

    * **coin** (*str*) - The coin of interest.
    * **dataframe** (*pd.DataFrame*) - The dataframe to process.
    * **price_label** (*str*) - The column header name for the price column.

    .. code-block:: python
        :name: to_sequences

        def to_sequences(data: pd.DataFrame, seq_len: int) -> np.array

    Converts the data to sequences.

    PARAMETERS:

    * **data** (*pd.DataFrame*) - The dataframe to process.
    * **seq_len** (*int*) - Length for the output sequences.

    RETURNS:
        The data converted to sequences.
    RETURN TYPE:
        numpy array

    .. code-block:: python
        :name: preprocess

        def preprocess(
            raw_data: pd.DataFrame,
            seq_len: int,
            test_split: float
        ) -> (np.array, np.array, np.array, np.array)

    Preprocesses and splits data into training and testing data for input to LSTM.

    PARAMETERS:

    * **raw_data** (*pd.DataFrame*) - The dataframe to process.
    * **seq_len** (*int*) - Length for the output sequences.
    * **test_split** (*float*) - Train-test fractional split.

    RETURNS:
        Training and testing data.

    RETURN TYPE:
        tuple of numpy arrays

    .. code-block:: python
        :name: train

        def train() -> None

    Trains the model with a mean-squared-error loss function and adam optimiser.

    .. code-block:: python
        :name: predict

        def predict(X: np.array) -> np.array

    Predicts future prices given the input data.

    PARAMETERS:

    * **X** (*np.array*) - Input data.

    RETURNS:
        Predicted future prices.

    RETURN TYPE:
        numpy array.

.. code-block:: python
    :name: predict

    def predict(X: np.array, coin: str) -> np.array


Predicts future prices given the input data.

PARAMETERS:

* **X** (*np.array*) - Input data.
* **coin** (*str*) - Coin of interest.

RETURNS:
    Predicted future prices.

RETURN TYPE:
    numpy array.

.. code-block:: python
    :name: predict

    def main(coin: str, filepath: str) -> None

Generate a price-based LSTM model and show a comparison graph between the actual and predicted prices
for the testing dataset.

PARAMETERS:

* **coin** (*str*) - Coin of interest.
* **filepath** (*str*) - Path to csv file with related data for coin.