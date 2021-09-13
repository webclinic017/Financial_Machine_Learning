"""
Name : get_data_stocks.py in Project: Financial_ML
Author : Simon Leiner
Date    : 07.05.2021
Description: Get the financial Data of stocks or dax, etc: yaho fiannce
"""

import json.decoder
import requests.exceptions
import yfinance as yf
from pandas_datareader import data as pdr
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
import warnings
import pandas as pd

warnings.filterwarnings(category=FutureWarning,action="ignore")

# See: https://de.finance.yahoo.com/

# use finance environment
yf.pdr_override()

# plotting
sns.set_theme()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def get_data(now,start,stock,plotting = True):

    """

    This function gets the price information of the wanted stock.

    :param now: datetime: Time now
    :param start: datetime: Starttime
    :param stock: string: Symbol of wanted stock
    :param plotting: boolean: wether to plot or not
    :return: pd.Series: Prices

    Bug: https://github.com/ranaroussi/yfinance/issues/363
    """

    print("-" * 10)

    # print function name
    print(f"Get data for stock: {stock}")
    print("-" * 10)

    # punishment
    starttime = 4
    punisher = 4
    limit = 31

    while 1:

        try:

            if starttime == limit:
                starttime = (limit - 1)

            # get the data as a pandas Series
            df = pdr.get_data_yahoo(stock, start, now)["Adj Close"]

        except json.decoder.JSONDecodeError:
            print("No connection to the internet")
            time.sleep(60)
            continue

        except requests.exceptions.ConnectionError:
            print("No connection to the internet")
            time.sleep(60)
            continue

        except Exception as e:
            print(e)
            print(f"Ups, Something went wrong :Time {now} :Stock : {stock}")
            starttime += 1
            time.sleep(punisher * starttime)
            continue

        break

    print("-" * 10)

    # plotting
    if plotting == True:

        # for printing only and in %
        df_ret = df.pct_change().dropna() * 100

        # plotting
        plt.subplot(2, 2, 1)
        plt.plot(df, label="Adj Close", color="goldenrod")
        plt.ylabel("Price in € or $:")
        plt.title(f"Adjusted Close of {stock} stock:")

        # plotting
        plt.subplot(2, 2, 2)
        plt.plot(df_ret, label="Returns", color="goldenrod")
        plt.ylabel("Returns in %:")
        plt.axhline(y=0, color="black")
        plt.title(f"Returns of {stock} stock:")

        # plotting
        plt.subplot(2, 2, 3)
        plt.title(f"Distribution of Adj Close:")
        sns.distplot(df, fit=norm)
        plt.xlabel("Price in € or $:")

        # plotting
        plt.subplot(2, 2, 4)
        plt.title(f"Distribution of Returns:")
        sns.distplot(df_ret, fit=norm)
        plt.xlabel("Returns in %:")
        plt.show()

    return df


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

