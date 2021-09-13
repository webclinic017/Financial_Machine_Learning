"""
Name : adjust_data_functions.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: Pca transformation of the given variables
"""

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings

# disable some warnings
warnings.filterwarnings(category=FutureWarning,action="ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def pca_analysis(data_cleaner):

    """

     This function computes a principal component analysis for dimensionality reduction.

     :param data_cleaner: list with 2 pd.DataFrame: data_cleaner
     :return: list with 2 pd.DataFrame: data_cleaner

     Linear dimensionality reduction using Singular Value Decomposition of
     the data to project it to a lower dimensional space. The input data is centered but not
     scaled for each feature before applying the SVD.

     For further Inforamtion See:
     # https: // stats.stackexchange.com / questions / 55718 / pca - and -the - train - test - split
     # https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
     # https://stackoverflow.com/questions/55441022/how-to-aply-the-same-pca-to-train-and-test-set
     # https: // towardsdatascience.com / pca - using - python - scikit - learn - e653f8989e60
    """

    # create scaler build_model
    scaler_model = RobustScaler()

    # create pca build_model
    # choose the minimum number of principal components such that 95% of the variance is retained.
    pca_model = PCA(.95)

    # for each dataframe do:
    for i in range(len(data_cleaner)):

        # get the df
        df = data_cleaner[i]

        #  get the X features
        X = df.drop(["t"], axis=1)

        # for the training data do
        if i == 0:

            # fit the scaler build_model
            scaler_model.fit(X)

            # fit the pca build_model
            pca_model.fit(X)

        # for the testing data do:
        else:
            pass

        # only transform the data with the already fitted scler build_model
        X = scaler_model.transform(X)

        # only transform the data with the already fitted pca build_model
        principal_components = pca_model.transform(X)

        # save them in a dataframe
        principal_df = pd.DataFrame(data=principal_components, index=df.index)

        # only print for the training data
        if i == 0:

            print(f"{principal_components.shape[1]} Principal components explain 95 % of the training datasets variance.")
            print("-" * 10)

        # add the y column
        finalDf = pd.concat([principal_df, df[['t']]], axis=1)

        # only plot for the training data
        if i == 0:

            # plotting
            plt.subplot(2, 1, 1)
            plt.title(f"First 2 Principal Components that explain the most variance:")
            sns.scatterplot(data=finalDf, x=finalDf.iloc[:,0], y=finalDf.iloc[:,1], hue=finalDf['t'],palette=["red","green"])
            plt.xlabel("PC1")
            plt.ylabel("PC2")

            # plotting
            plt.subplot(2, 1, 2)
            plt.title(f"Distribution of the PCA transformed returns:")
            sns.distplot(principal_df, fit=norm)
            plt.show()

        # set the data
        data_cleaner[i] = finalDf

    # return the data
    return data_cleaner

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def transform_supervies_learning(df,days_predict_into_future):

    """

    This function transforms a time series into a supervised learning problem.

    :param df: pd.DataFrame: information of Labels
    :param days_predict_into_future: integer: number of days to predict in the future
    :return: pd.DataFrame: Adjusted and transformed df
    """

    print(f"The time series containes {len(df.Label)} datapoints.")
    print("-" * 10)

    # rename a column
    df.rename({"Label": "t"}, inplace=True, axis=1)

    # number of days to look backward and convert into columns: make a 2 : 3 split
    numberdays_back = int(len(df["t"]) / 3)

    # transform the dataframe, so we can use him properly
    for i in range(days_predict_into_future, numberdays_back):
        df['t-' + str(i)] = df["t"].shift(i)

    # remove the nan values : delete many rows, because we shifted the infromation into the columns and the last row, because we have t+1
    df.dropna(inplace=True)

    # note: t and t+1 must also be accounted for
    print(f"Transforming the data into {numberdays_back+2} columns.")
    print("-" * 10)

    print(f"The dataframe contains {df.shape[0]} rows and {df.shape[1]} columns. A total of {df.shape[0] * df.shape[1]} datapoints.")
    print("-" * 10)

    return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
