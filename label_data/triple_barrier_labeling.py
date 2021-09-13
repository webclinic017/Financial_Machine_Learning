"""
Name : triple_barrier_labeling_old.py in Project: Financial_ML
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

def get_daily_vol(returns, span=100):

    """

    This function computes the exponential moving average standard deviation of the returns.

    :param returns: pd.Series: returns
    :param span: integer: timespan for the weighted moving average
    :return: pd.Series: daily volatitiy estimate
    """

    # get daily volatility: standard deviation of the returns
    daily_vol = returns.ewm(span=span).std()

    return daily_vol

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_cumsumfiltered(returns,h):

    """

    This function applies the CUMSUM Filter.

    :param returns: pd.Series: returns
    :param h: float: filter size
    :return: pd.DatetimeIndex: Timestamps of the surviving dates
    """

    # initial values: zero and some empty lists
    filtered_values, spos, sneg, counter_lost_datapoints = [], 0, 0, 0

    # loop through the index:
    for i in returns.index:

        # apply the formula of the symmetric Cumsum filter
        spos, sneg = max(0,spos + returns.loc[i]), min(0,sneg + returns.loc[i])

        if sneg < -h:

            # reset sneg
            sneg = 0

            # append the value to the "survival list"
            filtered_values.append(i)

        elif spos > h:

            # reset spos
            spos = 0

            # append the value to the "survival list"
            filtered_values.append(i)

        else:

            # count as we filtered out this datapoint
            counter_lost_datapoints += 1

    print(f"{counter_lost_datapoints} out of {len(returns)} datapoints have been filtered out ({round((counter_lost_datapoints/len(returns)) *100,2)} %) by the CUMSUM Filter.")
    print("-" * 10)

    return pd.DatetimeIndex(filtered_values)

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

def get_time_fist_touch(ps,factors,vertical_barriers,st_daily):

    """

    This function gets the time the first out of the three barriers was touched.

    :param ps: pd.Series: Prices
    :param factors: float: that sets the size of the 2 horizontal barriers
    :param vertical_barriers: pd.Series: containing the timestamps of the vertical barriers
    :param st_daily: pd.Series: daily volatitlity as unit width of the horizontal barriers
    :return: pd.DataFrame: Timestamps of the first barrier touched
    """

    # create a container to save the values
    barriers = pd.concat({"price":ps, "trgt": st_daily,"vert_barrier": vertical_barriers}, axis=1)

    # timestmaps at which each barrier is touched [upper_horz,lower_horz,vertical_barriers]
    df_three_barriers = get_timestamps_barriers_touched(ps = ps,barriers = barriers,factors = factors)

    # get the minimum: earliest time returned from barriers [upper_horz,lower_horz,vertical_barriers]
    # # Get a series containing minimum value of each row, all’ : If all values are NA, drop that row or column.
    barriers["barrier_touched_first"] = df_three_barriers.dropna(how = "all").min(axis = 1)

    # drop na rows along the given column
    barriers.dropna(subset = ["barrier_touched_first"],inplace = True)

    return barriers

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_timestamps_barriers_touched(ps,barriers,factors):

    """

    This function gets the time the first out of the three barriers was touched.

    :param ps: pd.Series: Prices
    :param barriers: pd.DataFrame: with all the needed information
    :param factors: float: that sets the size of the 2 horizontal barriers
    :return: pd.DataFrame: Timestamps when each of the three barriers was triggert
    """

    # df to write the barriers into with already having the column vertical_barriers
    df_three_barriers = barriers[["vert_barrier"]].copy(deep =True)

    # upper_horz as upper horizontal barrier
    if factors[0] > 0:

        # width of upper horizontal barrier
        upper_horz = factors[0] * barriers["trgt"]

    else:

        # series with nan values
        upper_horz = pd.Series(index = barriers.index)

    # lower_horz as lower horizontal barrier
    if factors[1] > 0:

        # width of lower horizontal barrier
        lower_horz = -factors[1] * barriers["trgt"]

    else:

        # series with nan values
        lower_horz = pd.Series(index=barriers.index)

    # timestamps of the vertical barrier. Last numdays nan values are filled with the most recent date of the Adjusted Close Series
    vert_barriers = barriers["vert_barrier"].fillna(ps.index[-1])

    # loop through the items of the data
    for index_day,vert_barrier_day in vert_barriers.iteritems():

        # prices from the index to the value of the horizontal barriers: 2 Dates : df with prices between the two dates
        df_help = ps[index_day:vert_barrier_day]

        # calculate the path returns : RETRUNS HERE, thus lower and upper horz are not added with the price information
        df_help = (df_help / ps[index_day] - 1)

        # write earliest lower horizontal barrier in column upper_horz, row after row: earliest stop loss
        df_three_barriers.loc[index_day, "upper_horz"] = df_help[df_help > upper_horz[index_day]].index.min()

        # write earliest upper horizontal barrier in column lower_horz, row after row: earliest profit taking
        df_three_barriers.loc[index_day, "lower_horz"] = df_help[df_help < lower_horz[index_day]].index.min()

    # only for writing the price horizontal barriers into the container
    barriers["top_barrier"] = ps + upper_horz
    barriers["bottom_barrier"] = ps + lower_horz

    # only for writing the new gained information in the container
    barriers["upper_horz"] = df_three_barriers["upper_horz"]
    barriers["lower_horz"] = df_three_barriers["lower_horz"]

    return df_three_barriers

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

    # get the index of the Df und merge them with the values of the colum that contains timestamps and drop the duplicates afterwards
    # timestaps from unionjoin between events index and barrier_touched_first timestamps: return differnet timestamps
    # px = barriers.index.union(barriers["barrier_touched_first"].values)
    # print(px)
    # print(ps)

    # px = px.drop_duplicates()

    # align the prices with the timestmaps defined by the px object:
    # px = ps.reindex(px, method="bfill")
    # print(px)

    # print(ps)
    # print(px.equals(ps))

    # calculate the path returns : differnece between price at starting date and date the price hit a barrier:
    df_label["ret"] = ps.loc[barriers["barrier_touched_first"].values].values / ps.loc[barriers.index] - 1

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

    # change the index to not the first, nut the horizontal barrier
    df_label.set_index(barriers["barrier_touched_first"], inplace=True)

    return df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_triple_barrier_labeld_data(ps,h,num_days,factors,plotting,cumsum):

    """

    This function combines all the above functions and executes them in the correct order

    :param ps: pd.Series: Prices
    :param h: float: filter size
    :param numdays: integer: number of days to add for vertical barrier
    :param factors: float: that sets the size of the 2 horizontal barriers
    :param plotting: boolean: wether to plot or not
    :param cumsum: boolean: wether to apply the cumsum filter or not
    :return: pd.DataFrame: Labels with the starting date of the triple barrier as index
    """

    # most recent datapoint
    most_recent_date = ps.index[-1].strftime('%Y-%m-%d')

    # deal with returns and Loose first Datapoint
    returns_ = ps.pct_change()

    if cumsum == True:

        # filtered values
        wanted_dates = get_cumsumfiltered(returns_, h)

        # subset the Series
        ps = ps[wanted_dates]

        # subset the Series
        returns_ = returns_[wanted_dates]

    else:
        pass

    # get daily volatility:
    st_daily = get_daily_vol(returns_)

    # get the vertical barriers
    vertical_barriers = get_vertical_barriers(ps, num_days)

    # most recent horizontal barrier
    most_recent_horizontal = pd.to_datetime(str(vertical_barriers.values[-1]))

    # get the Dataframe with all the needed information
    barriers = get_time_fist_touch(ps, factors=factors,vertical_barriers=vertical_barriers, st_daily=st_daily)

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
    print(barriers.head(5))
    print("-" * 10)
    print(barriers.tail(5))
    print("-" * 10)

    if plotting == True:

        # plotting
        plot_one_triple_barrier(barriers, ps, num_days,factors)

    # show the returning DF
    print(f"Overview of the Labeld Data:")
    print(df_label)
    print("-" * 10)

    return df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def plot_one_triple_barrier(barriers,ps,numdays,factors,last_barriers = 5):

    """

    This function plots some triple barrier events.

    :param barriers: pd.DataFrame: with all the needed information
    :param ps: pd.Series: Prices
    :param numdays: integer: number of days to add for vertical barrier
    :param factors: float: that sets the size of the 2 horizontal barriers
    :param last_barriers: integer: number on how many barriers to plot
    :return: None
    """

    # barriers old index
    starting_dates_tpb = barriers.index

    # change the index to not the first, but the horizontal barrier
    barriers.set_index("barrier_touched_first", inplace=True)

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

        # ith upper barrier
        upper_barrier = barriers["top_barrier"][-i]
        # print(upper_barrier)

        # ith lower barrier
        lower_barrier = barriers["bottom_barrier"][-i]
        # print(lower_barrier)

        # get the price
        price = plotting_price_recent_x_months[-i]

        # no horizontal barriers
        if factors[0] == 0 and factors[1] == 0:
            # plot

            plt.plot([start, end], [price, price], color=f"{colors[i]}", linestyle="--")
            plt.plot([start, start], [price - 0.1, price + 0.1], color=f"{colors[i]}", linestyle="-")
            plt.plot([end, end], [price - 0.1, price + 0.1], color=f"{colors[i]}", linestyle="-")

        # stop loss limits active: upper and lower barrier exist
        else:
            # plot
            plt.plot([start, end], [upper_barrier, upper_barrier], color=f"{colors[i]}", linestyle="--")
            plt.plot([start, end], [lower_barrier, lower_barrier], color=f"{colors[i]}", linestyle="--")
            plt.plot([start, start], [lower_barrier, upper_barrier], color=f"{colors[i]}", linestyle="-")
            plt.plot([end, end], [lower_barrier, upper_barrier], color=f"{colors[i]}", linestyle="-")

    # stylistical stuff
    plt.ylabel("Price in € or $:")
    plt.title(f"Adjusted Close and the last {last_barriers} TPBs from {last_x_months.strftime('%Y-%m-%d')} up to {plotting_price_recent_x_months.index[-1].strftime('%Y-%m-%d')}:")
    date_format = mdates.DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #