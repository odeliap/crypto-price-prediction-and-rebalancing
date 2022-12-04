"""
Sentiment Analyzer
"""

# ------------- Libraries -------------
import logging

from typing import List

import pandas as pd
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------

keep_columns = ['open', 'high', 'low', 'subjectivity', 'polarity', 'compound', 'negative', 'neutral',
                'positive', 'timestamp']

# Set filepaths
bitcoin_input_filepath = '../data_cleaner/outputs/bitcoin_dataset.csv'
ethereum_input_filepath = '../data_cleaner/outputs/ethereum_dataset.csv'
solana_input_filepath = '../data_cleaner/outputs/solana_dataset.csv'

filepaths = [bitcoin_input_filepath, ethereum_input_filepath, solana_input_filepath] # files to perform sentiment analysis on

# ------------- Class -------------

class SentimentAnalyzer:

    def __init__(
        self,
        dataframe: pd.DataFrame
    ) -> None:
        """
        Initializes a new sentiment analyzer object.

        Parameters
        __________
        dataframe : pandas dataframe
            Dataset to perform sentiment analysis on.
        """
        self.dataframe = dataframe

        self.dataframe['text'] = self.clean_headlines()

        self.dataframe['subjectivity'] = self.dataframe['text'].apply(self.get_subjectivity)
        self.dataframe['polarity'] = self.dataframe['text'].apply(self.get_polarity)

        self.daily_sentiment_score_retriever()

        self.dataframe = self.dataframe[keep_columns]


    def clean_headlines(self) -> List[str]:
        """
        Cleans headlines and returns list of cleaned headlines.

        Returns
        __________
        clean_headlines : list of strings
            Clean news headlines.
        """
        headlines = []
        clean_headlines = []

        if 'text' in self.dataframe.columns:
            headline_index = self.dataframe.columns.get_loc('text')
            for row in range(0, len(self.dataframe.index)):
                headlines.append(' '.join(str(x) for x in self.dataframe.iloc[row, headline_index]))
        else:
            raise Exception("'text' column not found in file; add 'text' column and rerun to proceed.")

        for i in range(0, len(headlines)):
            clean_headlines.append(re.sub("b[(')]", '', headlines[i])) # remove b'
            clean_headlines[i] = re.sub('b[(")]', '', clean_headlines[i]) # remove b"
            clean_headlines[i] = re.sub("\'", '', clean_headlines[i]) # remove \'

        logging.info('cleaned headers')

        return clean_headlines


    def daily_sentiment_score_retriever(self) -> None:
        """
        Update dataframe with columns for sentiment scores
        of compound, negative, neutral, and positive.
        """
        compound = []
        neg = []
        pos = []
        neu = []

        for i in range(0, len(self.dataframe['text'])):
            SIA = self.get_SIA(self.dataframe['text'][i])
            compound.append(SIA['compound'])
            neg.append(SIA['neg'])
            neu.append(SIA['neu'])
            pos.append(SIA['pos'])

        self.dataframe['compound'] = compound
        self.dataframe['negative'] = neg
        self.dataframe['neutral'] = neu
        self.dataframe['positive'] = pos


    @staticmethod
    def get_subjectivity(text: str) -> float:
        """
        Get subjectivity of text.

        Parameters
        __________
        text : string
            Text to get subjectivity for.

        Returns
        _______
        subjectivity_score : float
            Subjectivity score for inputted text.
        """
        logging.info('getting subjectivity')
        subjectivity_score = TextBlob(text).sentiment.subjectivity
        return subjectivity_score


    @staticmethod
    def get_polarity(text: str) -> float:
        """
        Get polarity of text.

        Parameters
        __________
        text : string
            Text to get polarity for.

        Returns
        _______
        polarity_score : float
            Polarity score for inputted text.
        """
        logging.info('getting polarity')
        polarity_score = TextBlob(text).sentiment.polarity
        return polarity_score


    @staticmethod
    def get_SIA(text: str) -> dict:
        """
        Get sentiment score of text.

        Parameters
        __________
        text : string
            Text to get sentiment score for.

        Returns
        _______
        sentiment : dictionary
            Sentiment scores for inputted text.
        """
        logging.info('getting SIA')
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return sentiment



if __name__ == "__main__":
    """
    Loop over files and perform sentiment analysis for each.
    """
    for file in filepaths:
        logging.info(f'processing file: {file}')
        filename = file[file.rfind('/')+1:]
        output_filename = filename.replace('dataset', 'sentiment_dataset')
        output_filepath = f'outputs/{output_filename}'
        dataframe = pd.read_csv(file)
        analyzer = SentimentAnalyzer(dataframe)
        analyzer.dataframe.to_csv(output_filepath, index=False)
        logging.info(f'finished processing {file}')
