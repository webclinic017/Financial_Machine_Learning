"""
Name :fixed_window_labeling.py in Project: Financial_ML
Author : Simon Leiner
Date    : 19.05.2021
Description: sheet contains the functions for computing the triple barrier method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# some printing options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_vertical_barriers(ps,numdays):

    """

    This function calculates the horizontal barriers numdays ahead.

    :param ps: pd.Series: Prices
    :param numdays: integer: number of days to add for vertical barrier
    :return: pd.Series: Timestamps of vertical barriers
    """

    # get the index as numbers and add numdays
    vertical_barriers = ps.index.searchsorted(ps.index + pd.Timedelta(days = numdays))

    # remove the last entries: by shifting we exceeded the number of rows
    vertical_barriers = vertical_barriers[vertical_barriers < ps.shape[0]]

    # subset the price Series index and get the index too
    vertical_barriers = pd.Series(ps.index[vertical_barriers], index = ps.index[:vertical_barriers.shape[0]])

    return vertical_barriers

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Checked: Functions work

def get_labels(barriers,ps):

    """

    This function calcualtes the path returns and thus the Labels for each triple barrier event.

    :param barriers: pd.DataFrame: with all the needed information
    :param ps: pd.Series: Prices
    :return: pd.DataFrame: Labels with the starting date of the triple barrier as index
    """

    # drop na rows if there isn't a horizontal barrier
    barriers.dropna(subset = ["vert_barrier"], inplace = True)

    # create a df with the events as index
    df_label = pd.DataFrame(index = barriers.index)

    # calculate the path returns : differnece between price at starting date and date the price hit a barrier:
    df_label["ret"] = ps.loc[barriers["vert_barrier"].values].values / ps.loc[barriers.index] - 1

    # add colum bin: indicate the sign of a number element-wise. returns 1 ,0, -1 for the sign
    df_label["Label"] = np.sign(df_label["ret"])

    # note if the return is exactly 0, np.sign return 0, as we want a binary problem, we have to convert these rare events to another class
    for index, value in enumerate(df_label["Label"]):
        if value == 0:

            # declare 0 returns as negative returns
            df_label["Label"][index] = -1

    # only for writing the new gained information in the container
    barriers["ret"] = df_label["ret"]
    barriers["Label"] = df_label["Label"]

    return df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_triple_barrier_labeld_data(ps,h,num_days,plotting = True):

    """

    This function combines all the above functions and executes them in the correct order

    :param ps: pd.Series: Prices
    :param h: float: filter size
    :param numdays: integer: number of days to add for vertical barrier
    :param plotting: boolean: wether to plot or not
    :return: pd.DataFrame: Labels with the starting date of the triple barrier as index
    """

    # most recent datapoint
    most_recent_date = ps.index[-1].strftime('%Y-%m-%d')

    # get the vertical barriers
    vertical_barriers = get_vertical_barriers(ps, num_days)

    # most recent horizontal barrier
    most_recent_horizontal = pd.to_datetime(str(vertical_barriers.values[-1]))

    # create a container to save the values
    barriers = pd.concat({"price": ps, "vert_barrier": vertical_barriers}, axis=1)

    # get the labeld Data
    df_label = get_labels(barriers,ps)

    # get the class distribution
    distribution = df_label["Label"].value_counts(normalize=True)

    # flag warning
    if (distribution[1] > 0.75) or (distribution[1] < 0.15):
        print(f"Be careful, the class distribution is very unbalanced with positive: {round(distribution[1],2)*100} % and negative returns: {round(distribution[2],2)*100} %")
        print("-" * 10)

    print(f"The last triple barrier for {num_days} days back has formed on the {df_label.index[-1].strftime('%Y-%m-%d')} - {most_recent_horizontal.strftime('%Y-%m-%d')} and there is data up to the {most_recent_date}.")
    print("-" * 10)

    # show the df with all the given information
    print(f"Overview of the triple barrier labling approach:")
    print(barriers.tail(2))
    print("-" * 10)

    if plotting == True:

        # plotting
        plot_one_triple_barrier(barriers, ps, num_days)

    # change the index to not the first, nut the horizontal barrier
    df_label.set_index(vertical_barriers,inplace = True)

    # show the returning DF
    # print(f"Overview of the Labeld Data:")
    # print(df_label.tail(3))
    # print("-" * 10)

    return df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def plot_one_triple_barrier(barriers,ps,numdays,last_barriers = 5):

    """

    This function plots some triple barrier events.

    :param barriers: pd.DataFrame: with all the needed information
    :param ps: pd.Series: Prices
    :param numdays: integer: number of days to add for vertical barrier
    :param last_barriers: integer: number on how many barriers to plot
    :return: None
    """

    # barriers old index
    starting_dates_tpb = barriers.index

    # change the index to not the first, but the horizontal barrier
    barriers.set_index("vert_barrier", inplace=True)

    # get the most recent date
    now = datetime.now()

    # last_barriers times the timepan of the horizontal barriers
    last_x_months = now - timedelta(days=31 + numdays)

    # plot the last formed triple barrier and just the most recent month
    plotting_price_recent_x_months = ps[(ps.index >= last_x_months) & (ps.index <= barriers.index[-1])]
    # print(plotting_price_recent_x_months)

    # plot the last Labled Datapoints (returns)
    plotting_labels_returns_recent_x_months = barriers[barriers.index >= last_x_months]["ret"] * 100
    # print(plotting_labels_returns_recent_x_months)

    # plotting
    plt.subplot(2, 3, 1)

    # plot the Labels of the last recent month
    plt.plot(plotting_labels_returns_recent_x_months,color="goldenrod")

    # stylistical stuff
    plt.ylabel("Returns in %:")
    plt.axhline(y=0, color="black")
    plt.title(f"Returns forming the Labels from {last_x_months.strftime('%Y-%m-%d')} up to {plotting_labels_returns_recent_x_months.index[-1].strftime('%Y-%m-%d')}:")
    date_format = mdates.DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    # plotting
    plt.subplot(2, 3, 4)

    # plot the adjusted Close of the last recent month
    plt.plot(plotting_price_recent_x_months, label="Adj Close", color="goldenrod")

    # possible colors to select
    all_colors = ["black","blue","teal","navy","grey","darkred","dimgray","royalblue","teal","cyan","maroon","indigo"]

    # define a colorrange:
    colors = []

    # create a colorlist
    for j in range(1,(last_barriers + 2)):

        # choose one color
        color = all_colors[j]

        # append the color
        colors.append(color)

    # loop over the last recent points
    for i in range(1,len(colors)):

        # ith last starting date
        start = starting_dates_tpb[-i]
        # print(start)

        # ith last vertical barrier
        end = barriers.index[-i]
        # print(end)

        price = plotting_price_recent_x_months[-i]

        # plot
        plt.plot([start, end], [price, price], color=f"{colors[i]}", linestyle="--")
        plt.plot([start, start], [price -0.1,price +0.1], color=f"{colors[i]}", linestyle="-")
        plt.plot([end, end], [price -0.1,price +0.1], color=f"{colors[i]}", linestyle="-")

    # stylistical stuff
    plt.ylabel("Price in € or $:")
    plt.title(f"Adjusted Close and the last {last_barriers} TPBs from {last_x_months.strftime('%Y-%m-%d')} up to {plotting_price_recent_x_months.index[-1].strftime('%Y-%m-%d')}:")
    date_format = mdates.DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #