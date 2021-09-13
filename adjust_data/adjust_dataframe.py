"""
Name : adjust_dataframe.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: Only a special Case of tbl with both factos being 0!
"""

from adjust_data import adjust_data_functions as ad
from sklearn.model_selection import train_test_split
from label_data import triple_barrier_labeling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import matplotlib.dates as mdates
import warnings

# disable some warnings
warnings.filterwarnings(category=FutureWarning,action="ignore")
pd.options.mode.chained_assignment = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def adjust_data(ps,factors,h ,numdays,days_predict_into_future ,pca,cumsum):

    """

    This function adjusts the data for a binary machine Learning Algorithm.

    :param ps: pd.Series: Prices
    :param factors: float: that sets the size of the 2 horizontal barriers
    :param h: float: filter size
    :param numdays: integer: number of days to add for vertical barrier
    :param days_predict_into_future: integer: number of days to predict in the future
    :param pca: boolean: apply pca transfomration
    :param cumsum: boolean: wether to apply the cumsum filter or not
    :return: list with 2 pd.DataFrame: data_cleaner, pd.DataFrame: df_label

    """

    # get the labeld Data
    df_label = triple_barrier_labeling.get_triple_barrier_labeld_data(ps, h, numdays, factors,True, cumsum)

    # get only the Labels
    df_only_label = df_label[["Label"]]

    # ensure numeric data
    df_only_label["Label"] = pd.to_numeric(df_only_label["Label"])

    # plotting
    plt.subplot(2, 3, 2)
    plt.plot(df_only_label["Label"].cumsum(), color="goldenrod")
    plt.ylabel("Label +1 or -1:")
    plt.title("Cumulative Triple-Barrier-Labeld DF:")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    # plotting
    plt.subplot(2, 3, 5)
    plt.title(f"Distribution of the Cumulative Labeld Data:")
    sns.distplot(df_only_label["Label"].cumsum(),fit = norm)
    plt.xlabel("Label +1 or -1:")

    # transform the time series into the wanted form
    df_only_label = ad.transform_supervies_learning(df_only_label,days_predict_into_future)
    # print(df_only_label)

    # plotting
    plt.subplot(2, 3, 3)
    plt.plot(df_only_label["t"].cumsum(), label="Cumulative transformed Labeld Data", color="goldenrod")
    plt.ylabel("Label +1 or -1:")
    plt.title("Cum. transformed supervised Learning DF (y-col):")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    # plotting
    plt.subplot(2, 3, 6)
    plt.title(f"Distribution of the Cum. supervised Learning Data:")
    sns.distplot(df_only_label["t"].cumsum(),fit = norm)
    plt.xlabel("Label +1 or -1:")
    plt.show()

    # divide the dataframe before applying transformations in order to avoid data leakage
    df_train, df_test = train_test_split(df_only_label, test_size=0.2, random_state=1,shuffle=False)

    # apply the transofmation to both dataframes seperately
    data_cleaner = [df_train, df_test]

    if pca == True:

        # apply PCA transformation: no need for that
        data_cleaner = ad.pca_analysis(data_cleaner)

    else:

        # do nothing
        pass

    # return the data
    return data_cleaner, df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #










