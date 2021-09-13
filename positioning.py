"""
Name : positioning.py in Project: Financial_ML
Author : Simon Leiner
Date    : 02.09.2021
Description: Put or Call options: buy or sell
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def choose_strategy_binary(predictions, true_values,strategy = "buy"):

    """

    This function calculates the +-1 binary correctly classified labels given a certain strategy.

    :param predictions: pd.Series: predictions from the model
    :param true_values: pd.Series: the true labeld values
    :param strategy: string: one of buy, sell, both
    :return: pd.Series: correctly classified directions
    """

    # make a deep copy of the predictions
    predictions_strategy = predictions.copy()

    # can't exploit the sell prediction
    if strategy == "buy":
        pass_chance_value = -1

    # can't exploit the buy prediction
    elif strategy == "sell":
        pass_chance_value = 1

    if strategy != "both":

        # adjust the predictions: don't earn money if label is -1 : pass the situation
        for index, value in predictions_strategy.items():

            # if value: prediction is pass_chance_value: leave that one out: set the value as 0
            if value == pass_chance_value:
                predictions_strategy[index] = 0

    # making money: 1*1, 1*0, -1*0, -1 * 1
    money_making = predictions_strategy * true_values
    # print(predictions_strategy)
    # print(true_values)
    # print(money_making)

    return money_making

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def choose_strategy_returns(ps,predictions, df_label, strategy = "buy"):

    """

    This function calculates the +-1 binary correctly classified labels given a certain strategy.

    :param ps: pd.Series: Prices
    :param predictions: predictions from the build_model
    :param df_label: df with the dates and returns and Labels calculated with the triple barrier approach
    :param strategy: one of buy, sell, both
    :return: pd.Series returns, pd.Series prices
    """

    # make a deep copy of the predictions
    predictions_strategy_binary = predictions.copy()

    # can't exploit the sell prediction
    if strategy == "buy":
        pass_chance_value = -1

    # can't exploit the buy prediction
    elif strategy == "sell":
        pass_chance_value = 1

    if strategy != "both":

        # adjust the predictions: don't earn money if label is -1 : pass the situation
        for index, value in predictions_strategy_binary.items():

            # if value: prediction is pass_chance_value: leave that one out: set the value as 0
            if value == pass_chance_value:
                predictions_strategy_binary[index] = 0

    # align the index with the predictions
    ps_comparison = ps[predictions_strategy_binary.index]

    # get the returns
    returns_comparison = ps_comparison.pct_change()
    # print(returns_comparison)

    # making money: quantative returns
    # money_making = df_label["ret"] * predictions_strategy_binary
    money_making = returns_comparison * predictions_strategy_binary
    # print(predictions_strategy_binary)
    # print(money_making)

    # drop na rows: df_label["ret"] is much longer then predictions_strategy_binary and * : NAN
    money_making.dropna(inplace = True)
    # print(money_making)

    return money_making, ps_comparison

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
