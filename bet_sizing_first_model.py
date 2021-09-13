"""
Name : bet_sizing.py in Project: Financial_ML
Author : Simon Leiner
Date    : 26.06.2021
Description: Define the size of the bet directly from predicted probabilities
"""

import pandas as pd
from scipy.stats import norm

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def adjust_predictions_by_threshold(y_predprob,threshold = 0.5):

    """This function adjusts the predictions by a given threshold so we are more precise on one class

    :param y_predprob: probabilities for different classes
    :param threshold: threshold value for positive predictions
    :return: predicted values

    """

    # empty list to save the predictions
    predictions = []

    # loop through predictions
    for x in y_predprob:

        # if probability for positive return bigger then ceratin threshold declare return as positive
        if x[1] > threshold:

            return_ = 1

        else:
            return_ = -1

        # append to list
        predictions.append(return_)

    return predictions

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def getSignal(prob_model,stepSize,pred_model):

    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    :param prob_model: probablility of the build_model at class 1
    :param stepSize: (float) The step size at which the bet size is discretized, default is 0.0 which imposes nodiscretization.
    :param pred_model: (pd.Series) The predicted bet side.
    :return: (pandas.Series) The bet size, with the time index.

    See: https://mlfinlab.readthedocs.io/en/latest/modelling/bet_sizing.html
    """

    # if there aren't any predictions
    if prob_model.shape[0] == 0:
        return pd.Series()

    # generate signals from the formula of the book

    # size from predicted probabilities
    signal0 = (prob_model - 1. / 2) / (prob_model * (1. - prob_model)) ** 0.5
    # plt.plot(signal0)
    # plt.show()

    # signal=side*size : 0 if side  = 0: second build_model said first build_model failed with classificating: unsecure
    signal0 = pred_model * (2 * norm.cdf(signal0) - 1)
    # plt.plot(signal0)
    # plt.show()

    # discrete the values
    signal1 = discreteSignal(signal0=signal0, stepSize=stepSize)
    # print(signal1)
    # plt.plot(signal1)
    # plt.show()

    return signal1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #