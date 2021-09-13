"""
Name : backtesting.py in Project: Financial_ML
Author : Simon Leiner
Date    : 02.09.2021
Description: backtest the Strategy

See: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
"""

import bt
from datetime import datetime
from get_data import get_data_stocks, STOCKS
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import matplotlib

now = datetime.now()
# get the wanted starttime
start = STOCKS.start
stock = "BAS.DE"
"ORCL,LHA.DE"
data = pdr.get_data_yahoo(stock, start, now)[["Adj Close"]]

matplotlib.use('TkAgg')

# use finance environment
yf.pdr_override()

# calculate moving average DataFrame using pandas' rolling_mean
import pandas as pd
# a rolling mean is a moving average, right?
sma = data.rolling(50).mean()

# let's see what the data looks like - this is by no means a pretty chart, but it does the job
plot = bt.merge(data, sma).plot(figsize=(15, 5))
plt.show()

class SelectWhere(bt.Algo):

    """
    Selects securities based on an indicator DataFrame.

    Selects securities where the value is True on the current date (target.now).

    Args:
        * signal (DataFrame): DataFrame containing the signal (boolean DataFrame)

    Sets:
        * selected

    """
    def __init__(self, signal):
        self.signal = signal

    def __call__(self, target):
        # get signal on target.now
        if target.now in self.signal.index:
            sig = self.signal.loc[target.now]

            # get indices where true as list
            selected = list(sig.index[sig])

            # save in temp - this will be used by the weighing algo
            target.temp['selected'] = selected

        # return True because we want to keep on moving down the stack
        return True

signal = data > sma

# first we create the Strategy
s = bt.Strategy('above50sma', [SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

# now we create the Backtest
t = bt.Backtest(s, data)

# and let's run it!
res = bt.run(t)

# what does the equity curve look like?
res.plot()
plt.show()

# and some performance stats
res.display()

# and some performance stats
res.display_lookback_returns()

# ok and how does the return distribution look like?
res.plot_histogram()
plt.show()