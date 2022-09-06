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

KEEP_COLUMNS = ['open', 'high', 'low', 'volume', 'subjectivity', 'polarity', 'compound', 'negative', 'neutral',
                'positive', 'price']

FILEPATH = '' # TODO: put correct filepath
OUTPUT_FILEPATH = '' # TODO: put correct output filepath

# ------------- Class -------------

class SentimentAnalyzer:

    def __init__(self, filepath):
        """
        Initializes a new sentiment analyzer object.

        :param filepath: path to dataset to perform sentiment analysis on
        :type str
        """
        self.filepath = filepath

        self.dataframe = pd.read_csv(filepath)

        self.dataframe['headlines'] = self.clean_headlines()

        self.dataframe['subjectivity'] = self.dataframe['headlines'].apply(self.get_subjectivity)
        self.dataframe['polarity'] = self.dataframe['headlines'].apply(self.get_polarity)

        self.daily_sentiment_score_retriever()

        self.dataframe = self.dataframe[KEEP_COLUMNS]


    def clean_headlines(self) -> List[str]:
        """
        Cleans headlines and returns list of cleaned headlines.

        :return: clean_headlines: list of clean headlines
        :rtype: List[str]
        """
        headlines = []
        clean_headlines = []

        if 'headlines' in self.dataframe.columns:
            headline_index = self.dataframe.columns.get_loc('headlines')
            for row in range(0, len(self.dataframe.index)):
                headlines.append(' '.join(str(x) for x in self.dataframe.iloc[row, headline_index]))
        else:
            raise Exception("'headlines' column not found in file; add 'headlines' column and rerun to proceed.")

        for i in range(0, len(headlines)):
            clean_headlines.append(re.sub("b[(')]", '', headlines[i])) # remove b'
            clean_headlines[i] = re.sub('b[(")]', '', clean_headlines[i]) # remove b"
            clean_headlines[i] = re.sub("\'", '', clean_headlines[i]) # remove \'

        return clean_headlines


    def daily_sentiment_score_retriever(self):
        compound = []
        neg = []
        pos = []
        neu = []

        for i in range(0, len(self.dataframe['headlines'])):
            SIA = self.get_SIA(self.dataframe['headlines'][i])
            compound.append(SIA['compound'])
            neg.append(SIA['neg'])
            neu.append(SIA['neu'])
            pos.append(SIA['pos'])

        self.dataframe['compound'] = compound
        self.dataframe['negative'] = neg
        self.dataframe['neutral'] = neu
        self.dataframe['positive'] = pos


    @staticmethod
    def get_subjectivity(text):
        """
        Get subjectivity of text.

        :param text: text to get subjectivity for
        :type str

        :return: subjectivity score for inputted text
        :rtype: float
        """
        return TextBlob(text).sentiment.subjectivity


    @staticmethod
    def get_polarity(text):
        """
        Get polarity of text.

        :param text: text to get polarity for
        :type str

        :return: polarity score for inputted text
        :rtype float
        """
        return TextBlob(text).sentiment.polarity


    @staticmethod
    def get_SIA(text):
        """
        Get sentiment score of text.

        :param text: text to get sentiment score for
        :type str

        :return: sentiment: Python dictionary of sentiment scores for inputted text
        :rtype dict[float]
        """
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return sentiment



if __name__ == "__main__":
    analyzer = SentimentAnalyzer(FILEPATH)
    analyzer.dataframe.to_csv(OUTPUT_FILEPATH)
