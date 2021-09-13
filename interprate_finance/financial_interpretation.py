"""
Name : financial_interpretation.py in Project: Financial_ML
Author : Simon Leiner
Date    : 02.09.2021
Description: Interpret the predictions
"""

import bet_sizing_first_model
from build_model import model_building
import positioning
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import matplotlib.dates as mdates

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def inverse_pct_change(initial_stock_value,returns):

    """

    This function compute stock prices froma series of returns and an initial value

    :param initial_stock_value: float: initial stock price
    :param returns: pd.Series: Series of returns
    :return: pd.Series: Series of stock price
    """

    # if not already done: drop na values from the return Series
    returns.dropna(inplace = True)
    # print(returns)

    # list for saving the values
    list_save_values = []

    # loop through the DF
    for index, value in returns.iteritems():

        # calculate the stock price
        val_new = initial_stock_value + initial_stock_value * value

        # append the new calculated value to the list
        list_save_values.append(val_new)

        # set the new initial value as the recently calculated value : always reinvesting case: zinseszinseffekt
        initial_stock_value = val_new

    # convert the list with the saved values to a pandas series and adjust the index
    ps = pd.Series(list_save_values, index=returns.index)

    return ps

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def financial_interprate_model(ps,y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test,df_label):

    """

    This function is still under construction

    :param ps: pd.Series: Prices
    :param y_pred_cv: CV predictions: +-1
    :param y_pred_test: Test predictions: +-1
    :param y_prob_cv: CV predictions of probabilities
    :param y_prob_test: Test predictions of probabilities
    :param x_train: X values of the training data
    :param x_test: X values of the testing data
    :param y_train: y values of the training data
    :param y_test: y values of the testing data
    :param df_label: dataframe with returns and Labels from the triple barrier method
    :return: None
    """

    # define the strategy
    strategy_buy = "buy"

    # define an comparison strategy
    strategy_both = "both"

    # append the training and testing predictions, true values and predicted probabilities
    predictions, true_values, predictions_prob, train_end_date = model_building.compute_predictions_true_values(y_pred_cv, y_pred_test, y_prob_cv, y_prob_test, x_train, x_test, y_train, y_test)

    # returns money making buys
    money_making_returns_buy, ps_comparison = positioning.choose_strategy_returns(ps,predictions, df_label, strategy = strategy_buy)

    # returns money making both
    money_making_returns_both, ps_comparison = positioning.choose_strategy_returns(ps, predictions, df_label,strategy=strategy_both)

    # plotting
    plt.subplot(2, 2, 1)
    plt.plot(money_making_returns_buy * 100, label="Correct Classified", color="black")
    plt.axvline(x=train_end_date, color="navy", label="Train Data End")
    plt.axhline(y=0, color="black")
    plt.ylabel("Returns in %:")
    plt.title(f"Exploitation of the Correct Side prediction with strategy {strategy_buy}:")
    plt.legend(loc="best")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    # plotting
    plt.subplot(2, 2, 3)
    plt.plot(money_making_returns_both * 100, label="Correct Classified", color="black")
    plt.axvline(x=train_end_date, color="navy", label="Train Data End")
    plt.axhline(y=0, color="black")
    plt.ylabel("Returns in %:")
    plt.title(f"Exploitation of the Correct Side prediction with strategy {strategy_both}:")
    plt.legend(loc="best")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    # prices money making buy
    money_making_prices_buy = inverse_pct_change(initial_stock_value=ps_comparison[0], returns= money_making_returns_buy)

    # prices money making both
    money_making_prices_both = inverse_pct_change(initial_stock_value=ps_comparison[0], returns=money_making_returns_both)

    # plotting
    plt.subplot(2, 2, 2)
    plt.plot(money_making_prices_buy, label="Money by Model", color="black")
    plt.plot(ps_comparison, label="Adjusted Close", color="goldenrod")
    plt.axvline(x=train_end_date, color="navy", label="Train Data End")
    plt.ylabel("Price in € or $:")
    plt.title(f"Exploitation of the Correct Side prediction with strategy {strategy_buy}:")
    plt.legend(loc="best")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)

    # plotting
    plt.subplot(2, 2, 4)
    plt.plot(money_making_prices_both, label="Money by Model", color="black")
    plt.plot(ps_comparison, label="Adjusted Close", color="goldenrod")
    plt.axvline(x=train_end_date, color="navy", label="Train Data End")
    plt.ylabel("Price in € or $:")
    plt.title(f"Exploitation of the Correct Side prediction with strategy {strategy_both}:")
    plt.legend(loc="best")
    date_format = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.show()



    # bet sizing: final side and size prediction
    # final_predictions = bet_sizing_first_model.getSignal(predictions_prob, 0.01, predictions)
    # print(final_predictions)

    # start index of the predictions
    # predection_starting_date = final_predictions.index[0]
    # print(predection_starting_date)

    #df for plotting
    # df_plotting = df[df.index >= predection_starting_date]
    # print(df_plotting)

    # starting value of the stock
    # stock_starting_value = df[df.index == predection_starting_date].values
    # print(stock_starting_value)

    # cumulative final predictions and changed first value
    # final_predictions_plotting = final_predictions.copy()
    # final_predictions_plotting.iloc[0] = final_predictions_plotting.iloc[0] + stock_starting_value

    # plotting, note predictions must be -1,1 and the true values also
    # plt.plot((predictions * true_value_fist_model).cumsum(), label="Gain / Loss +- 1", color="goldenrod")
    # plt.plot((final_predictions * true_value_fist_model).cumsum(), label="Gain / Loss (-1,1)",color="black")
    # plt.axvline(x=predictions_train.index[-1], color="navy", label="Train Data End")
    # plt.ylabel("Correct classified predictions +1 or -1:")
    # plt.title(f"Correct vs False classified predictions +1 or -1:")
    # plt.legend(loc="best")
    # plt.show()

    # plotting
    # plt.subplot(2, 1, 2)
    # plt.title(f"Distribution of the Cumulative Predicted Values vs. Cumulative True Values:")
    # sns.distplot(final_predictions.cumsum(), fit=norm, label="Predictions")
    # sns.distplot(true_value_fist_model.cumsum(), label="True Values", color="dimgray")
    # plt.legend(loc="best")
    # plt.show()



    return None