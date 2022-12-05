"""
Unpickler to load models and scalers

Made to fix a bug loading pickled objects using multiple modules with the help of:
https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
"""

# ----------- Libraries -----------
import pickle


# ----------- Class -----------

class Unpickler(pickle.Unpickler):
    """
    Class for unpickling objects.
    """

    def find_class(self, module, name):
        if name == 'SentimentPriceLSTM':
            from src.models.SentimentPriceLSTMModel import SentimentPriceLSTM
            return SentimentPriceLSTM
        """elif name == 'SentimentLSTMModel':
            from src.models.SentimentLSTMModel import SentimentLSTMModel
            return SentimentLSTMModel
        elif name == 'SampleSentimentModel':
            from src.models.SampleSentimentModel import SampleSentimentModel
            return SampleSentimentModel
        elif name == 'PriceLSTMModel':
            from src.models.PriceLSTMModel import PriceLSTMModel
            return PriceLSTMModel"""
        return super().find_class(module, name)


# ----------- Methods -----------

def load_object(fully_qualified_filepath: str):
    """
    Load object from path.

    Parameters
    __________
    fully_qualified_filepath : string
        File path to load object from.

    Returns
    _______
    unpickled_object
        Unpickled object.
    """
    unpickled_object = Unpickler(open(fully_qualified_filepath, 'rb')).load()
    return unpickled_object
