"""
Name : triple_barrier_labeling_old.py in Project: Financial_ML
Author : Simon Leiner
Date    : 19.05.2021
Description: sheet contains the functions for computing the triple barrier method
"""

import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_daily_vol(returns, span=100):

    """
    :param returns: a pandas Series of returns
    :param span: timespan for the weighted moving average
    :return: daily volatitiy estimate
    """

    # get daily volatility
    daily_vol = returns.ewm(span=span).std()
    # print(daily_vol)

    # print(f"Data corresponding to the dates {daily_vol[daily_vol.isna() == True].index.values} has been removed (Volatility).")
    # print("-" * 10)

    # drop na values : Loose 2 Datapoint at the beginning
    # daily_vol.dropna(inplace=True)

    # print(daily_vol)
    return daily_vol

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_cumsumfiltered(returns,h):

    """
    CUSUM Filter :
    :param returns: a pandas Series of returns
    :param h: threshold , filter size
    :return: pandas timeindex with timestamps for which we will create every triple barrier : Less Data then Before

    We Are loosing our first Datapoint here and potentially all that get filtered out
    """

    # initial values
    filtered_values, spos, sneg, counter_lost_datapoints = [], 0, 0, 0

    # loop through the index
    for i in returns.index:

        # apply the formuly of the symmetric Cumsum filter
        spos, sneg = max(0,spos + returns.loc[i]), min(0,sneg + returns.loc[i])

        # 0 or the price : cumulative
        # print(spos)

        # 0 or the price : cumulative
        # print(sneg)

        # date
        # print(i)

        if sneg < -h:

            # reset sneg
            sneg = 0

            filtered_values.append(i)

        elif spos > h:

            # reset spos
            spos = 0
            filtered_values.append(i)

        else:

            # count as we filtered out this datapoint
            counter_lost_datapoints += 1

    print(f"{counter_lost_datapoints} out of {len(returns)} datapoints have been filtered out ({round((counter_lost_datapoints/len(returns)) *100,2)} %) by the CUMSUM Filter.")
    print("-" * 10)

    # print(filtered_values)

    # return it a s a datetime index : potentially loose x datapoints
    return pd.DatetimeIndex(filtered_values)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_vertical_barriers(ps,numdays):

    """
    This function simply sets the backward looking of each triple barrier : 1 day : 1 day shift of timestamps

    :param ps: a pandas Series of prices
    :param numdays: number of days to add for vertical barrier
    :return: pd.SeriesTimestamps of vertical barriers: time shifted by numdays from filtered_values
    """

    # get the index as number  day ahead
    vertical_barriers = ps.index.searchsorted(ps.index + pd.Timedelta(days = numdays))
    # print(vertical_barriers)
    # print(ps.index)
    # print(ps.index + pd.Timedelta(days = numdays))

    # remove the last entries: by shifting we exceeded the number of rows in close
    vertical_barriers = vertical_barriers[vertical_barriers < ps.shape[0]]
    # print(vertical_barriers)

    # pd Series with values: timestamps at t1 and index: timestamps: all rows, nb of rows in t1
    vertical_barriers = pd.Series(ps.index[vertical_barriers], index = ps.index[:vertical_barriers.shape[0]])

    # print(vertical_barriers)
    return vertical_barriers

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_time_fist_touch(ps,factors,vertical_barriers,st_daily):

    """
    :param ps: a pandas Series of Prices
    :param factors: non negative float that sets the size of the 2 horizontal barriers
    :param vertical_barriers: pandas series containing the timestamps of the vertical barriers
    :param st_daily: unit width of the horizontal barriers

    Case 1: side not in barriers: bin in (-1,1) :label by price information
    Case 2: side in barriers: bin in (0,1) :label by meta-labeling

    :return: time of the first barrier touch
    """

    # create wanted output: with infromation needed to find the timestamps the barriers get touched
    events = pd.concat({"price":ps,"vertical_barriers": vertical_barriers, "trgt": st_daily}, axis=1)
    #.dropna(subset = ["trgt"])
    # print(events)

    # timestmaps at which each barrier is touched [upper_horz,lower_horz,vertical_barriers]
    df_help,upper_horz, lower_horz = get_timestamps_barriers_touched(ps = ps,events = events,factors = factors)
    # print(df_help)

    # get the minimum: earliest time returned from barriers [upper_horz,lower_horz,vertical_barriers]
    # # Get a series containing minimum value of each row
    events["barrier_touched_first"] = df_help.dropna(how="all").min(axis = 1)
    # print(events)

    # dataframe with columns vertical_barriers: timestamp of the first barrier that was touched and trgt: target
    return events,upper_horz, lower_horz

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_timestamps_barriers_touched(ps,events,factors):

    """ This function computes the 3 barriers

    :param ps: a pandas Series of Prices
    :param events: Dataframe with columns index, vertical_barriers, st_daily, side
    :param factors: List of 2 float values that multipy trgt to set the width od the upper and lower barrier
    :return:
    """

    # df to write the barriers into with alrady having the column vertical_barriers
    df_barrier_touched = events[["vertical_barriers"]].copy(deep =True)
    # print(df_barrier_touched)

    # pt as upper horizontal barrier
    if factors[0] > 0:

        # width of upper horizontal barrier
        upper_horz = factors[0] * events["trgt"]
        # print(upper_horz)

    else:

        # series with nan values
        upper_horz = pd.Series(index = events.index)

    # sl as lower horizontal barrier
    if factors[1] > 0:

        # width of lower horizontal barrier
        lower_horz = -factors[1] * events["trgt"]
    else:

        # series with nan values
        lower_horz = pd.Series(index=events.index)

    # timestamps of the vertical barrier with filled na values
    data_loop = events["vertical_barriers"].fillna(ps.index[-1])

    # loop through the items of the data
    for index,value in data_loop.iteritems():

        # prices from the index to the value of the horizontal barriers: 2 Dates : df with prices between the two dates
        df_help = ps[index:value]

        # calculate the path returns
        df_help = (df_help / ps[index] - 1)

        # write earliest lower horizontal barrier in column upper_horz, row after row: earliest stop loss
        df_barrier_touched.loc[index, "upper_horz"] = df_help[df_help > upper_horz[index]].index.min()

        # write earliest upper horizontal barrier in column lower_horz, row after row: earliest profit taking
        df_barrier_touched.loc[index, "lower_horz"] = df_help[df_help < lower_horz[index]].index.min()

    # dataframe containing the timestamps at which each barrier was touched [upper_horz,lower_horz,vertical_barriers]
    # print(df_barrier_touched)
    return df_barrier_touched,upper_horz, lower_horz

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_labels(events,ps):

    """
    :param events: time of the first barrier touch
    :param ps: a pandas Series of Prices

    :return: (Meta) labeled events

    Note: events is an DF with:

    index as events starttime
    vertical_barriers as events endtime
    st_daily as events target

    """

    # print(f"Data corresponding to the dates {events.loc[pd.isna(events.barrier_touched_first), :].index} has been removed. (Didn't touch any barrier yet)")
    # print("-" * 10)

    # drop na values from the pandas series containing the timestamps of the vertical barriers
    events.dropna(subset = ["barrier_touched_first"],inplace = True)
    # print(events["barrier_touched_first"])

    # get the index of the Df und merge them with the values of the colum that contains timestamps and drop the duplicates afterwards
    # timestaps from unionjoin between events index and barrier_touched_first timestamps: return differnet timestamps
    px = events.index.union(events["barrier_touched_first"].values).drop_duplicates()
    # print(px)

    # align the prices with the timestmaps defined by the px object:
    px = ps.reindex(px,method = "bfill")

    # create a df with the events as index
    df_out = pd.DataFrame(index = events.index)
    # print(df_out)

    # calculate the path returns : differnece between price at starting date and date the price hit a barrier:
    df_out["ret"] = px.loc[events["barrier_touched_first"].values].values / px.loc[events.index] - 1

    # add colum bin: indicate the sign of a number element-wise. returns 1 ,0, -1 for the sign
    df_out["Label"] = np.sign(df_out["ret"])
    # print(df_out["Label"])

    # note if the return is exactly 0, np.sign return 0, as we want a binary problem, we have to convert these rare events to another class
    for index, value in enumerate(df_out["Label"]):
        if value == 0:

            # declase 0 returns as negative returns
            df_out["Label"][index] = -1

    # print(df_out)
    return df_out

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def get_triple_barrier_labeld_data(ps,h,num_days,factors):

    """

    :param ps: pandas series containg the price information
    :param h: filter parameter for Cumsum filter: the smaller the value, the less filtered
    :param num_days: number of days for the horizontal barriers
    :param factors: non negative float that sets the size of the 2 horizontal barriers
    :return:
    """

    # most recent datapoint
    most_recent_date = ps.index[-1].strftime('%Y-%m-%d')

    # print function name
    # print(f"Execute Labeling function : {inspect.currentframe().f_code.co_name}")
    # print("-" * 10)

    # deal with returns and Loose first Datapoint
    returns_ = ps.pct_change().dropna()

    # print(f"Data corresponding to the dates {returns_[returns_.isna() == True].index.values} has been removed (Returns).")
    # print("-" * 10)

    # filtered values (Loose potentional x Datapoints)
    wanted_dates = get_cumsumfiltered(returns_, h)
    # print(wanted_dates)

    # subset the Series
    ps = ps[wanted_dates]
    # print(ps)

    # subset the Series
    returns_ = returns_[wanted_dates]
    # print(ps)

    # get daily volatility : Loose one additional Datapoint
    st_daily = get_daily_vol(returns_)
    # print(st_daily)

    # get the vertical barriers
    vertical_barriers = get_vertical_barriers(ps, num_days)

    # most recent horizontal barrier
    most_recent_horizontal = pd.to_datetime(str(vertical_barriers.values[-1]))

    # get the "events" : time of the first barrier touched
    events,upper_horz, lower_horz = get_time_fist_touch(ps, factors=factors,vertical_barriers=vertical_barriers, st_daily=st_daily)

    # get the labeld Data
    labeld = get_labels(events,ps)

    # get the class distribution
    distribution = labeld["Label"].value_counts(normalize=True)

    # flag warning
    if (distribution[1] > 0.75) or (distribution[1] < 0.15):
        print(f"Be careful, the class distribution is very unbalanced with positive: {round(distribution[1],2)*100} % and negative returns: {round(distribution[2],2)*100} %")
        print("-" * 10)

    print(f"The last triple barrier for {num_days} days back has formed on the {labeld.index[-1].strftime('%Y-%m-%d')} - {most_recent_horizontal.strftime('%Y-%m-%d')} and there is data up to the {most_recent_date}.")
    print("-" * 10)

    # create a df with all the given information
    df_all_combined = pd.concat({"price":ps, "trgt": st_daily,"upper_barrier": ps + upper_horz,"lower_barrier": ps + lower_horz,"vertical_barrier": vertical_barriers, "returns": labeld["ret"],"label":labeld["Label"]}, axis=1).dropna(axis = 0)
    print(f" Overview of the triple barrier labling approach:")
    print(df_all_combined.tail(10))
    print("-" * 10)

    # plotting
    plot_one_triple_barrier(df_all_combined,ps)

    # print(labeld.index)
    # print(vertical_barriers)

    #change the index to not the first, nut the horizontal barrier
    # labeld.set_index(vertical_barriers,inplace = True)

    # print(labeld)
    return labeld

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Functions work

def plot_one_triple_barrier(df_all_combined,ps,last_barriers = 5):

    """
    :param df_all_combined: df with all information
    :param ps: price information
    :param last_barriers: number on how many barriers to draw
    :return: None
    """

    # get the most recent date
    now = datetime.now()

    # last month date
    last_month = now - timedelta(days=31)

    # plot the last formed triple barrier and just the most recent month
    plotting_price_recent_month = ps[ps.index > last_month]
    # print(plotting_price_recent_month)

    # plotting
    plt.subplot(2, 3, 1)

    # plot the adjusted Close of the last recent month
    plt.plot(plotting_price_recent_month, label="Adj Close", color="goldenrod")

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
        start = df_all_combined.index[-i]
        # print(start)

        # ith last vertical barrier
        end = df_all_combined["vertical_barrier"][-i]
        # print(end)

        # ith upper barrier
        upper_barrier = df_all_combined["upper_barrier"][-i]
        # print(upper_barrier)

        # ith lower barrier
        lower_barrier = df_all_combined["lower_barrier"][-i]
        # print(lower_barrier)

        # plot
        plt.plot([start, end], [upper_barrier, upper_barrier], color = f"{colors[i]}", linestyle = "--")
        plt.plot([start, end], [lower_barrier, lower_barrier], color =f"{colors[i]}", linestyle = "--")
        plt.plot([start, start], [lower_barrier, upper_barrier], color =f"{colors[i]}", linestyle = "-")
        plt.plot([end, end], [lower_barrier, upper_barrier], color =f"{colors[i]}", linestyle = "-")

    # stylistical stuff
    plt.ylabel("Price in € or $:")
    plt.title(f"Adjusted Close from {last_month.strftime('%Y-%m-%d')} up to {now.strftime('%Y-%m-%d')}:")
    date_format = mdates.DateFormatter('%m-%d')
    plt.gca().xaxis.set_major_formatter(date_format)
    # plt.show()

    return None