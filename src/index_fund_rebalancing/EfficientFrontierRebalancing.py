"""
Efficient Frontier Rebalancing Algorithm for index fund rebalancing.

Efficient Frontier Rebalancing: weights based on expected future returns.

Made by following tutorial:
https://intrinio.medium.com/how-to-rebalance-your-stock-portfolio-with-python-71a188d70087
"""

# ------------- Libraries -------------

import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

# ------------- Class -------------

class EfficientFrontierRebalancing:

    def __init__(self, predicted_price_dataframe: pd.DataFrame):
        # Calculate mean predicted returns & standard deviation of returns
        mu = mean_historical_return(predicted_price_dataframe)
        S = CovarianceShrinkage(predicted_price_dataframe).ledoit_wolf()

        # Find optimal portfolio weightings that lie on the Efficient Frontier Curve
        eff_frontier = EfficientFrontier(mu, S)
        eff_frontier_weights = eff_frontier.max_sharpe()
        eff_frontier_clean_weights = eff_frontier.clean_weights()

