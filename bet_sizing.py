"""
Name : bet_sizing.py in Project: Financial_ML
Author : Simon Leiner
Date    : 26.06.2021
Description: Define the size of the bet directly from predicted probabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def adjust_predictions_by_threshold(y_predprob,threshold = 0.5,meta = None):

    """This function adjusts the predictions by a given threshold so we are more precide on one class

    :param y_predprob: probabilities for different classes
    :param threshold: threshold value for positive predictions
    :param meta: if meta Labeling is active
    :return: predicted values

    We want to be sure, that we predict the the positive returns very accurate
    """

    # we want to achieve a high recall with the first build_model -> lower threshold
    if meta is None:

        # adjust the threshold
        threshold = threshold #- 0.4

        # safety:
        if threshold <= 0.2:

            threshold = 0.2

        print(f"The new adjusted threshold in order to gain high recall is: {round(threshold, 2)}.")
        print("-" * 10)

    else:
        pass

    # empty list to save the predictions
    predictions = []

    # loop through predictions
    for x in y_predprob:

        # if probability for positive return bigger then ceratin threshold declare return as positive
        if x[1] > threshold:

            return_ = 1

        else:

            if meta is None:
                return_ = -1

            else:
                return_ = 0

        # append to list
        predictions.append(return_)

    return predictions

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def getSignal(pred_first_model,stepSize,prob_second_model,pred_second_model,horizontal_barriers = None):

    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    :param horizontal_barriers: the expiry datetime of the product, with a datetime index, the datetime the position was taken.
    :param prob: (pandas.Series) The predicted probability.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size (i.e. without multiplying by the side).
    :param step_size: (float) The step size at which the bet size is discretized, default is 0.0 which imposes nodiscretization.
    :return: (pandas.Series) The bet size, with the time index.

    See: https://mlfinlab.readthedocs.io/en/latest/modelling/bet_sizing.html
    """

    # if there aren't any predictions
    if prob_second_model.shape[0] == 0:
        return pd.Series()

    # generate signals from the formula of the book

    # size from predicted probabilities
    signal0 = (prob_second_model - 1. / 2) / (prob_second_model * (1. - prob_second_model)) ** 0.5

    # signal=side*size : 0 if side  = 0: second build_model said first build_model failed with classificating: unsecure
    signal0 = pred_second_model * (2 * norm.cdf(signal0) - 1)

    # signal= positive or negative return predicted by the first build_model
    signal0 *= pred_first_model
    # print(signal0)

    signal1 = discreteSignal(signal0=signal0, stepSize=stepSize)
    # print(signal1)

    return signal1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def discreteSignal(signal0,stepSize):

    """This function discretizies the size of the trade

    :param signal0:
    :param stepSize: (0,1]: the size of the steps, the degree of discretization
    :return:

    Note: helps to reduce unnecessary overtrading"""

    # after formula proposed in the book
    signal1 = (signal0 / stepSize).round() * stepSize

    # cap
    signal1[signal1 > 1] = 1

    # floor
    signal1[signal1 < -1] = -1

    return signal1
