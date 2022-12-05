"""
Model index fund performance based on predictions for comparison against new index fund.

Made by following tutorial:
https://python.plainenglish.io/how-to-improve-investment-portfolio-with-rebalancing-strategy-in-python-a58841ee8b5e
"""

# ------------ Libraries ------------

import logging

import pandas as pd
import numpy as np
from datetime import datetime

import yfinance as yf

from Evaluation import plot_comparison, report_evaluation_metrics

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------ Constants ------------

# Coins available for rebalancing
coins = ['bitcoin', 'ethereum', 'solana']

num_stocks = 3  # Set number of stocks to hold in the portfolio
num_review = 1  # Set number of stocks to review

start_date = '2022-05-1'  # Set start date
end_date = '2022-07-1'  # Set end date

# Format start and end dates to strings of format yyyy-MM-dd
datetime_start = datetime.strptime(start_date, '%Y-%m-%d')
datetime_end = datetime.strptime(end_date, '%Y-%m-%d')

comparison_fund_name = 'BITW'

# ------------ Logic ------------

def portfolio(data: pd.DataFrame, num_stocks: int, num_review: int) -> pd.DataFrame:
    """
    Create a portfolio.

    Parameters
    ----------
    data : pandas dataframe
        Dataset of coins to corresponding stock price (USD) indexed by date.
    num_stocks : int
        Number of stocks to have in the fund.
    num_review : int
        Number of "bad stocks" to remove

    Returns
    -------
    return_dataframe : pandas dataframe
        Dataset of average monthly returns.
    """
    dataframe = data.copy()
    selected_stocks = []
    average_monthly_return = [0]

    for i in range(len(dataframe)):
        if len(selected_stocks) > 0:
            average_monthly_return.append(dataframe[selected_stocks].iloc[i, :].mean())
            bad_stocks = dataframe[selected_stocks].iloc[i, :].sort_values(ascending=True)[:num_review].index.values.tolist()
            selected_stocks = [t for t in selected_stocks if t not in bad_stocks]
        fill = num_stocks - len(selected_stocks)
        new_picks = dataframe.iloc[i, :].sort_values(ascending=False)[:fill].index.values.tolist()
        selected_stocks = selected_stocks + new_picks
    return_dataframe = pd.DataFrame(np.array(average_monthly_return), columns=["monthly_returns"])
    return return_dataframe


def main(stock_data: pd.DataFrame, num_stocks: int, num_review: int) -> pd.DataFrame:
    """
    Create balanced portfolio.

    Parameters
    ----------
    stock_data : pandas dataframe
        Input dataframe with stock data.
        Must be indexed on date and have columns of
        coin names corresponding to stock price in USD.
    num_stocks : int
        Number of stocks to have in the fund.
    num_review : int
        Number of "bad stocks" to remove

    Returns
    -------
    rebalanced_portfolio : pandas dataframe
        Rebalanced portfolio dataset.
    """
    stock_returns = pd.DataFrame()

    for coin in coins:
        stock_returns[coin] = stock_data['Adj Close'][coin].pct_change()

    stock_returns = stock_returns.dropna()

    rebalanced_portfolio = portfolio(stock_returns, num_stocks, num_review)
    return rebalanced_portfolio


if __name__ == "__main__":
    """
    Create rebalanced portfolio and compare against comparison index fund.
    """

    # Download comparison fund data and get monthly returns
    comparison_fund = yf.download(comparison_fund_name, start=start_date, end=end_date, internal='1d')
    comparison_fund["monthly_returns"] = comparison_fund["Adj Close"].pct_change().fillna(0)

    # Download stock data for available coins
    stock_data = yf.download(coins,start=start_date, end=end_date,interval='1d')
    stock_data = stock_data.dropna()

    # Get rebalanced portfolio for coins' stock data
    rebalanced_portfolio = main(stock_data, num_stocks, num_review)

    # Get and display evaluation metrics for rebalanced portfolio and comparison fund
    report_evaluation_metrics(rebalanced_portfolio, datetime_start, datetime_end)
    report_evaluation_metrics(comparison_fund, datetime_start, datetime_end, f"{comparison_fund_name} Index")

    # Plot rebalanced portfolio performance against comparison fund performance
    standard_fund_results = comparison_fund["monthly_returns"].reset_index(drop=True)
    plot_comparison(rebalanced_portfolio, standard_fund_results, comparison_fund_name)