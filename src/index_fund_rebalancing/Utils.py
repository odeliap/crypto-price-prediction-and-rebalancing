"""
Utils for index fund rebalancing evaluation
"""

# ----------- Libraries -----------

import pandas as pd
import numpy as np

import datetime

# ----------- Functions -----------

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