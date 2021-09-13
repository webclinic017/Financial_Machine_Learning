"""
Name : feature_selection_functions.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: feature selection functions
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest,f_classif, chi2, mutual_info_classif
from sklearn.linear_model import Lasso

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def uni_feature_selection(X,y,number_select, method = None):

    """

    This function selects k best features, given a scoring function, eg F-value, X^2, Anova, mutual information.

    Select features according to the k highest scores
    :param X: pd.DataFrame: DF with the training data: X-data: Labels at time t-x
    :param y: pd.Series: DF with the training data: y-column: Labels at time t
    :param number_select: integer: number of best columns to select
    :param method: string: scoring method
    :return: list: names of selected columns

    Note: F-value is linear, while mutual information can capture nonlinear relationships
    """

    # select the classification method
    if method == "F":
        score_function = f_classif
    elif method == "C":
        score_function = chi2
    else:
        score_function = mutual_info_classif

    #create a selector build_model
    selector = SelectKBest(score_function, k = number_select)

    #fit the selector and transform your data
    X_k_best_new = selector.fit_transform(X,y)

    #get the originally selected features: Need to reverse the transformation
    selected_features = pd.DataFrame(selector.inverse_transform(X_k_best_new),index = X.index, columns = X.columns)

    #get the selected columns
    selected_columns = selected_features.columns[selected_features.var() != 0]

    # print(f"Uni feature selection suggests to keep the columns: {selected_columns}.")
    # print("-" * 10)

    # return the selected columns
    return selected_columns

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def lasso_selection(X,y):

    """

    This function performs the lasso feature selection.

    :param X: pd.DataFrame: DF with the training data: X-data: Labels at time t-x
    :param y: pd.Series: DF with the training data: y-column: Labels at time t
    :return: list: selected columns

    Lasso linear build_model with iterative fitting along a regularization path. See glossary entry for cross-validation estimator.
    The best build_model is selected by cross-validation.
    The optimization objective for Lasso is: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    """

    # create a Lasso regression build_model with Cross validation for validation
    reg = Lasso(tol = 1e-3)
    # n_alphas int, default=100
    # fit_intercept bool, default=True
    # normalize bool, default=False

    #fit your build_model to the Data
    reg.fit(X, y)

    #get the fitted coefficients of the trained build_model
    coef = pd.Series(reg.coef_, index = X.columns)

    columns_to_keep = coef[coef != 0].index

    # print(f"Lasso regression suggests to keep the columns: {columns_to_keep}.")
    # print("-" * 10)

    # return the selected columns
    return columns_to_keep

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




