"""
Model index fund performance based on predictions.

Made by following tutorial:
https://python.plainenglish.io/how-to-improve-investment-portfolio-with-rebalancing-strategy-in-python-a58841ee8b5e
"""

# ------------ Libraries ------------

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import yfinance as yf

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------ Constants ------------

# Coins available for rebalancing
#coins = ['bitcoin', 'ethereum', 'solana']
coins = ['BTC', 'BCH', 'LTC', 'EOS', 'NEO', 'DASH', 'TRX']

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


def CAGR(data: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
    """
    Calculate and return compound annual growth rate.

    Parameters
    ----------
    data : pandas dataframe
        Rebalanced portfolio dataset.
    start_date : datetime date
        Start date for tracking prices.
    end_date : datetime date
        End date for tracking prices.

    Returns
    -------
    cagr : float
        Compound annual growth rate.
    """
    dataframe = data.copy()
    dataframe['cumulative_returns'] = (1 + dataframe['monthly_returns']).cumprod()
    trading_days = end_date - start_date
    n = len(dataframe) / trading_days.days
    cagr = (dataframe['cumulative_returns'][len(dataframe)-1])**(1/n) - 1
    return cagr


def volatility(data: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
    """
    Calculate and return the volatility.

    Parameters
    ----------
    data : pandas dataframe
        Rebalanced portfolio dataset.
    start_date : datetime date
        Start date for tracking prices.
    end_date : datetime date
        End date for tracking prices.

    Returns
    -------
    vol : float
        Volatility.
    """
    df = data.copy()
    trading_days = end_date - start_date
    vol = df['monthly_returns'].std() * np.sqrt(trading_days.days)
    return vol


def sharpe_ratio(data: pd.DataFrame, rf: float, start_date: datetime, end_date: datetime) -> float:
    """
    Calculate and return the sharpe ratio.

    The sharpe ratio measures the performance of an investment against a risk-free asset
    after adjusting for risk.

    Parameters
    ----------
    data : pandas dataframe
        Rebalanced portfolio dataset.
    rf : float
        Best available rate of return of a risk-free asset.
    start_date : datetime date
        Start date for tracking prices.
    end_date : datetime date
        End date for tracking prices.

    Returns
    -------
    sharpe : float
        Sharpe ratio.
    """
    dataframe = data.copy()
    sharpe = (CAGR(data, start_date, end_date) - rf) / volatility(dataframe, start_date, end_date)
    return sharpe


def maximum_drawdown(data: pd.DataFrame) -> float:
    """
    Calculate and return the maximum drawdown.

    The drawdown is the measure of decline from historical peak.

    Parameters
    ----------
    data : pandas dataframe
        Rebalanced portfolio dataset.

    Returns
    -------
    max_dd : float
        Maximum drawdown.
    """
    dataframe = data.copy()
    dataframe['cumulative_returns'] =  (1 + dataframe['monthly_returns']).cumprod()
    dataframe['cumulative_max'] = dataframe['cumulative_returns'].cummax()
    dataframe['drawdown'] = dataframe['cumulative_max'] - dataframe['cumulative_returns']
    dataframe['drawdown_pct'] = dataframe['drawdown'] / dataframe['cumulative_max']
    max_dd = dataframe['drawdown_pct'].max()
    return max_dd


def main(stock_data: pd.DataFrame, num_stocks: int, num_review: int) -> (pd.DataFrame, pd.DataFrame):
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
    stock_returns : pandas dataframe
        Stock returns.
    """
    stock_returns = pd.DataFrame()

    for coin in coins:
        stock_returns[coin] = stock_data['Adj Close'][coin].pct_change()

    stock_returns = stock_returns.dropna()

    rebalanced_portfolio = portfolio(stock_returns, num_stocks, num_review)
    return rebalanced_portfolio, stock_returns


if __name__ == "__main__":
    """
    Create rebalanced portfolio and compare against BITW index fund.
    """
    start_date = '2022-05-1' # Set start date
    end_date = '2022-07-1' # Set end date

    num_stocks = 3 # Set number of stocks to hold in the portfolio
    num_review = 1 # Set number of stocks to review

    # Download BITW fund data and get monthly returns
    bitw_fund = yf.download("BITW", start=start_date, end=end_date, internal='1d')
    bitw_fund["monthly_returns"] = bitw_fund["Adj Close"].pct_change().fillna(0)

    # Download stock data for available coins
    stock_data = yf.download(coins,start=start_date, end=end_date,interval='1d')
    stock_data = stock_data.dropna()

    # Get rebalanced portfolio for coins' stock data
    rebalanced_portfolio, stock_returns = main(stock_data, num_stocks, num_review)

    # Format start and end dates to strings of format yyyy-MM-dd
    datetime_start = datetime.strptime(start_date, '%Y-%m-%d')
    datetime_end = datetime.strptime(end_date, '%Y-%m-%d')

    # Get and display evaluation metrics for rebalanced portfolio
    logging.info("Rebalanced Portfolio Performance")
    logging.info("CAGR: " + str(CAGR(rebalanced_portfolio, datetime_start, datetime_end)))
    logging.info("Sharpe Ratio: " + str(sharpe_ratio(rebalanced_portfolio, 0.03, datetime_start, datetime_end)))
    logging.info("Maximum Drawdown: " + str(maximum_drawdown(rebalanced_portfolio)))

    logging.info(" ")

    # Get and display evaluation metrics for BITW fund
    logging.info("BITW Index Performance")
    logging.info("CAGR: " + str(CAGR(bitw_fund, datetime_start, datetime_end)))
    logging.info("Sharpe Ratio: " + str(sharpe_ratio(bitw_fund, 0.03, datetime_start, datetime_end)))
    logging.info("Maximum Drawdown: " + str(maximum_drawdown(bitw_fund)))

    # Plot rebalanced portfolio performance against BITW fund performance
    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots()
    plt.plot((1 + portfolio(stock_returns, num_stocks, num_review)).cumprod())
    plt.plot((1 + bitw_fund["monthly_returns"].reset_index(drop=True)).cumprod())
    plt.title("BITW Index Return vs Rebalancing Strategy Return")
    plt.ylabel("cumulative return")
    plt.xlabel("months")
    ax.legend(["Strategy Return", "Index Return"])
    plt.savefig(f"outputs/graphs/index_return_vs_rebalancing_strategy.png", dpi=300)
    plt.show()