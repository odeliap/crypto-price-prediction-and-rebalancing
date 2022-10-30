"""
Class to model index fund performance based on predictions.

Made by following tutorial:
https://python.plainenglish.io/how-to-improve-investment-portfolio-with-rebalancing-strategy-in-python-a58841ee8b5e

Link to yfinance library:
https://pypi.org/project/yfinance/

Look into:
https://intrinio.medium.com/how-to-rebalance-your-stock-portfolio-with-python-71a188d70087
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

#coins = ['bitcoin', 'ethereum', 'solana']
coins = ['BTC', 'BCH', 'LTC', 'EOS', 'NEO', 'DASH', 'TRX']

# ------------ Logic ------------

def portfolio(data, numStocks, numRev):
    """
    :param data: dataframe of coins to corresponding stock price (USD) indexed by date
    :type: pd.DataFrame

    :param numStocks: number of stocks in fund
    :type: int

    :param numRev: number of "bad stocks" to remove
    :type: int

    :return return_dataframe: average monthly return
    :rtype: pd.DataFrame
    """
    dataframe = data.copy()
    selected_stocks = []
    average_monthly_return = [0]

    for i in range(len(dataframe)):
        if len(selected_stocks) > 0:
            average_monthly_return.append(dataframe[selected_stocks].iloc[i, :].mean())
            bad_stocks = dataframe[selected_stocks].iloc[i, :].sort_values(ascending=True)[:numRev].index.values.tolist()
            selected_stocks = [t for t in selected_stocks if t not in bad_stocks]
        fill = numStocks - len(selected_stocks)
        new_picks = dataframe.iloc[i, :].sort_values(ascending=False)[:fill].index.values.tolist()
        selected_stocks = selected_stocks + new_picks
    return_dataframe = pd.DataFrame(np.array(average_monthly_return), columns=["monthly_returns"])
    return return_dataframe

def CAGR(data, start_date: datetime, end_date: datetime):
    dataframe = data.copy()
    dataframe['cumulative_returns'] = (1 + dataframe['monthly_returns']).cumprod()
    trading_days = end_date - start_date
    n = len(dataframe) / trading_days.days
    cagr = (dataframe['cumulative_returns'][len(dataframe)-1])**(1/n) - 1
    return cagr

def volatility(data, start_date: datetime, end_date: datetime):
    df = data.copy()
    trading_days = end_date - start_date
    vol = df['monthly_returns'].std() * np.sqrt(trading_days.days)
    return vol

def sharpe_ratio(data, rf, start_date, end_date):
    dataframe = data.copy()
    sharpe = (CAGR(data, start_date, end_date) - rf) / volatility(dataframe, start_date, end_date)
    return sharpe

def maximum_drawdown(data):
    dataframe = data.copy()
    dataframe['cumulative_returns'] =  (1 + dataframe['monthly_returns']).cumprod()
    dataframe['cumulative_max'] = dataframe['cumulative_returns'].cummax()
    dataframe['drawdown'] = dataframe['cumulative_max'] - dataframe['cumulative_returns']
    dataframe['drawdown_pct'] = dataframe['drawdown'] / dataframe['cumulative_max']
    max_dd = dataframe['drawdown_pct'].max()
    return max_dd


def main(stock_data: pd.DataFrame, start_date: str, end_date: str, numStocks: int, numRev: int):
    """
    :param stock_data: input dataframe, must be indexed on date and have columns of
    coin names corresponding to stock price in USD.
    :type: pd.DataFrame

    :param start_date:
    """
    bitw_fund = yf.download("BITW", start=start_date, end=end_date, internal='1d')
    bitw_fund["monthly_returns"] = bitw_fund["Adj Close"].pct_change().fillna(0)

    stock_returns = pd.DataFrame()

    for coin in coins:
        stock_returns[coin] = stock_data['Adj Close'][coin].pct_change()

    stock_returns = stock_returns.dropna()

    datetime_start = datetime.strptime(start_date, '%Y-%m-%d')
    datetime_end = datetime.strptime(end_date, '%Y-%m-%d')

    rebalanced_portfolio = portfolio(stock_returns, numStocks, numRev)
    logging.info("Rebalanced Portfolio Performance")
    logging.info("CAGR: " + str(CAGR(rebalanced_portfolio, datetime_start, datetime_end)))
    logging.info("Sharpe Ratio: " + str(sharpe_ratio(rebalanced_portfolio, 0.03, datetime_start, datetime_end)))
    logging.info("Maximum Drawdown: " + str(maximum_drawdown(rebalanced_portfolio)))

    logging.info(" ")

    logging.info("S&P500 Index Performance")
    logging.info("CAGR: " + str(CAGR(bitw_fund, datetime_start, datetime_end)))
    logging.info("Sharpe Ratio: " + str(sharpe_ratio(bitw_fund, 0.03, datetime_start, datetime_end)))
    logging.info("Maximum Drawdown: " + str(maximum_drawdown(bitw_fund)))

    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots()
    plt.plot((1 + portfolio(stock_returns, numStocks, numRev)).cumprod())
    plt.plot((1 + bitw_fund["monthly_returns"].reset_index(drop=True)).cumprod())
    plt.title("BITW Index Return vs Rebalancing Strategy Return")
    plt.ylabel("cumulative return")
    plt.xlabel("months")
    ax.legend(["Strategy Return", "Index Return"])
    plt.savefig(f"outputs/graphs/index_return_vs_rebalancing_strategy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    tickers = ['BTC', 'BCH', 'LTC', 'EOS', 'NEO', 'DASH', 'TRX']

    stock_data = yf.download(tickers,start='2022-05-1', end='2022-07-1',interval='1d')
    stock_data = stock_data.dropna()

    main(stock_data, '2022-05-1', '2022-07-1', 3, 1)