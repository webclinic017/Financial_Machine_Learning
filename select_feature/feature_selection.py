"""
Name : feature selection.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: feature selection by grouping the differenet methods
"""

from select_feature import feature_selection_functions as an

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def select_wanted_cols(df,number_cols):

    """
    This function selects the wanted columns by given metrics

    :param df: pd.DataFrame: DF with the training data
    :param number_cols: integer: fraction of columns to select
    :return: list: selected columns by feature selection

    Note Correlation only finds linear dependency, but mutual_info_classif goes for nonlinear dependencys
    """

    # get the x and y columns
    X = df.drop(["t"], axis=1)
    y = df["t"]

    # empty list to save our results
    wanted_columns = []

    # number of selecting columns : make a 1 : 3 split
    number_select = int(len(X.columns) / number_cols)

    # number_select cols with the highest correlation
    cols = df.corr().nlargest(number_select, "t")["t"].index

    # convert the resulst into a list
    cols_by_corr = list(cols.values)

    # note won't found y in X so remove it
    if "t" in cols_by_corr:
        cols_by_corr.remove("t")

    # Select the features by F-test
    selected_cols_uni_feature = an.uni_feature_selection(X, y, number_select, "F")

    # convert the resulst into a list
    cols_by_uni = list(selected_cols_uni_feature.values)

    # Select the features by Lasso regression
    selected_cols_lasso = an.lasso_selection(X, y)

    # convert the resulst into a list
    cols_by_lasso = list(selected_cols_lasso.values)

    # print(f"Correlation suggests to keep the columns: {cols_by_corr}.")
    # print("-" * 10)

    # add them all up
    wanted_columns = wanted_columns + cols_by_corr + cols_by_uni + cols_by_lasso + ["t"]

    # remove duplicates from list
    wanted_columns = list(dict.fromkeys(wanted_columns))

    print(f"Feature Selection extracted {len(wanted_columns)} out of {df.shape[1]} columns.")
    print("-" * 10)

    # return the data
    return wanted_columns

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



