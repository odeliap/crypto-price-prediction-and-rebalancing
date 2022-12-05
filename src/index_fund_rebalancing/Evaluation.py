"""
Methods for evaluation of index fund with visualization.
"""

# ------------ Libraries ------------

import logging

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from Utils import CAGR, sharpe_ratio, maximum_drawdown

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------ Functions ------------


def plot_comparison(new_fund: pd.DataFrame, standard_fund: pd.DataFrame, standard_fund_name: str) -> None:
    """
    Plot performance of funds in comparison.

    Parameters
    __________
    new_fund : pandas dataframe
        Index fund results from new trading algorithm.
    standard_fund : pandas dataframe
        Index fund results from standard trading algorithm.
    standard_fund_name : str
        Standard fund's name.
    """
    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots()
    plt.plot((1 + new_fund).cumprod())
    plt.plot((1 + standard_fund).cumprod())
    plt.title(f"{standard_fund_name} Index Return vs Rebalancing Strategy Return")
    plt.ylabel("cumulative return")
    plt.xlabel("months")
    ax.legend(["Strategy Return", "Index Return"])
    plt.savefig(f"outputs/graphs/index_return_vs_rebalancing_strategy.png", dpi=300)
    plt.show()


def report_evaluation_metrics(
        portfolio: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        portfolio_name: str = "Rebalanced Portfolio",
) -> None:
    """
    Report evaluation metrics.

    Parameters
    __________
    portfolio : pandas dataframe
        Rebalanced portfolio.
    start_date : datetime object
        Date stock returns started from.
    end_date : datetime object
        Date stock returns go to.
    portfolio_name : string, default = "Rebalanced Portfolio"
        Name of the rebalanced portfolio.
    """
    logging.info(f"{portfolio_name} Performance")
    logging.info("CAGR: " + str(CAGR(portfolio, start_date, end_date)))
    logging.info("Sharpe Ratio: " + str(sharpe_ratio(portfolio, 0.03, start_date, end_date)))
    logging.info("Maximum Drawdown: " + str(maximum_drawdown(portfolio)) + "\n")
