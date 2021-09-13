"""
Name : build_first_model.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description:  create the second build_model with meta- labeld, binary data to define the sizes of the bet
"""

# See: https://hudsonthames.org/meta-labeling-a-toy-example/, Labeling in Chapter 2

from adjust_data import adjust_dataframe
import validation_functions
from sklearn.model_selection import cross_val_predict
from build_first_model import compute_first_model
import pandas as pd
from datetime import datetime
from getdata import get_data, STOCKS
import bet_sizing
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# printing options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO: Not finished yet
# now
now = datetime.now()
# get the wanted starttime
start = STOCKS.start
stock = "ORCL"
df = get_data.get_data(now, start, stock)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

# See: https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html

def compute_second_model(df,h,number_days,pca,number_cols,n_splits,pct_emb):

    """ This function computes the first build_model"""

    # get the cv, and the predictions from the first build_model
    cv, predictions_fist_model,true_value_fist_model,df_previous,wanted_columns = compute_first_model(df, factors=[1.96, 1.96], h=h, number_days=number_days, pca=pca,number_cols = number_cols, n_splits=n_splits, pct_emb=pct_emb)

    print("Finished with the first build_model.")
    print("-" * 10)

    # get the finished Dataframes
    data_cleaner = adjust_dataframe.adjust_data(df, [1.96,1.96], h, number_days, pca, predictions_fist_model,df_previous)

    # training set
    df_train = data_cleaner[0]

    # testing set
    df_test = data_cleaner[1]

    # subset X by selected columns
    df_train = df_train[wanted_columns]
    df_test = df_test[wanted_columns]

    # divide the train Dataframe in X and Y
    x_train = df_train.drop(["t+1"], axis=1)
    y_train = df_train["t+1"]

    # divide the test Dataframe in X and Y
    x_test = df_test.drop(["t+1"], axis=1)
    y_test = df_test["t+1"]

    # scoring function: in the first build_model, we want to achieve a high recall / precision
    scoring = "f1"

    # compare the models
    best_model = validation_functions.algorithmncomp(x_train, y_train, cv, scoring)

    # tune the hyper parameters of the best build_model
    # best_model = parameter_tuning_functions.get_hyper_tuning(best_model, x_train, y_train, cv, scoring)

    print("Finished with Hyperparameter Tuning of the second build_model.")
    print("-" * 10)

    # compare score by scoring function with cross validation
    validation_functions.score_by_cv(best_model, x_train, y_train, cv, scoring)

    # make predictions with cross validation
    y_predprob_cv = cross_val_predict(best_model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=10)

    # plot the roc curve and get the auc score: get the best threshold lvl
    threshold = validation_functions.plot_roc_curve_cv(best_model, cv, x_train, y_train)
    # threshold = 0.5

    # adjusted predictions
    y_pred_adjusted_cv = bet_sizing.adjust_predictions_by_threshold(y_predprob_cv, threshold, meta=True)

    # get the confusion matrix, showing true and predicted values with cross validation
    validation_functions.get_confusionmatrix(y_train, y_pred_adjusted_cv,second_model=True)

    # evaluation by classreport with cross validation
    validation_functions.classreport(y_train, y_pred_adjusted_cv,second_model=True)

    # plot the learning curve
    validation_functions.plot_learning_curve(best_model, x_train, y_train, cv, scoring)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # predictions with final test set
    y_pred_test = best_model.predict(x_test)

    # predictions with final test set
    y_predprob_test = best_model.predict_proba(x_test)

    # get the optimal threshoold value
    # threshold = parameter_tuning_functions.get_optimal_threshold(y_pred_test,y_test)

    # plot the roc curve and get the auc score
    # validation_functions.plot_roc_curve_test(best_model, x_test, y_test, y_pred_test)

    # adjusted predictions
    # y_pred_adjusted = bet_sizing.adjust_predictions_by_threshold(y_predprob_test, threshold)

    # get the confusion matrix, showing true and predicted values with the test set
    # validation_functions.get_confusionmatrix(y_test, y_pred_adjusted)

    # evaluation by classreport with the test set
    # validation_functions.classreport(y_test, y_pred_adjusted)

    # plot the roc curve and get the auc score
    # validation_functions.plot_roc_curve_test(best_model, x_test, y_test, y_pred_adjusted)

    # convert the final predictions to a pandas Series and append the prices

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # convert the values to a numpy array
    y_train_array = np.array(y_train.values).astype(int)
    y_test_array = np.array(y_test.values).astype(int)
    # print(y_train_array)

    # convert the values to a numpy array
    y_predprob_cv = np.array(y_predprob_cv)
    y_predprob_test = np.array(y_predprob_test)
    # print(y_predprob_cv)

    correct_confidences_cv = y_predprob_cv[range(len(y_train_array)), y_train_array]
    correct_confidences_test = y_predprob_test[range(len(y_test_array)), y_test_array]
    # print(correct_confidences_cv)

    # convert the values to a pandas Series
    predictions_train_prob = pd.Series(correct_confidences_cv, index=x_train.index)
    predictions_test_prob = pd.Series(correct_confidences_test, index=x_test.index)

    # append the predictions together
    predictions_prob = predictions_train_prob.append(predictions_test_prob)

    # convert the final predictions to a pandas Series and append the prices
    predictions_train = pd.Series(y_pred_adjusted_cv, index=x_train.index)
    predictions_test = pd.Series(y_pred_test, index=x_test.index)

    # append the predictions together
    predictions = predictions_train.append(predictions_test)
    # print(predictions)

    # true values
    true_value_second_model = y_train.append(y_test)
    # print(true_value_fist_model)

    # class distribution of the predictions
    distrib_pred = predictions.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_pred)

    # class distribution of the predictions
    distrib_true = true_value_second_model.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_true)

    print("Class distribution of the predictions vs the true values (training and testing datasets) by the second build_model:")
    print(f"Correctly classified datapoints  : Predicted: {round(distrib_pred.iloc[1, 0], 4) * 100} % vs. True {round(distrib_true.iloc[1, 0], 4) * 100} %")
    print(f"False classified datapoints      : Predicted: {round(distrib_pred.iloc[0, 0], 4) * 100} % vs. True {round(distrib_true.iloc[0, 0], 4) * 100} %")
    print("-" * 10)

    # for plotting
    predictions_plotting = predictions.copy()
    true_value_second_model_plotting = true_value_second_model.copy()

    # adjust the values for plotting: 0,1 -> -1, 1: Only for plotting!!
    for index, value in predictions_plotting.items():

        if value == 0:
            predictions_plotting[index] = -1

    # adjust the values for plotting: 0,1 -> -1, 1: Only for plotting!!
    for index, value in true_value_second_model_plotting.items():

        if value == 0:
            true_value_second_model_plotting[index] = -1

    # plotting
    plt.subplot(2, 1, 1)
    plt.plot(predictions_plotting.cumsum(), label="Prediction", color="black")
    plt.plot(true_value_second_model_plotting.cumsum(), label="True Value", color="goldenrod")
    plt.axvline(x=predictions_train.index[-1], color="navy", label="Train Data End")
    plt.ylabel("Predicted Values:")
    plt.xlabel("True Values:")
    plt.title(f"Cumulative Predictions vs. True values of the one step ahead triple-barrier-label with a span of {number_days} days:")
    plt.legend(loc="best")

    plt.subplot(2, 2, 3)
    plt.title(f"Distribution of the Cumulative Predicted Values :")
    sns.distplot(predictions_plotting.cumsum(), fit=norm)

    plt.subplot(2, 2, 4)
    plt.title(f"Distribution of the Cumulative True Values:")
    sns.distplot(true_value_second_model_plotting.cumsum(), fit=norm)
    plt.show()

    # bet sizing: final side and size prediction
    final_predictions = bet_sizing.getSignal(predictions_fist_model, 0.01, predictions_prob, predictions, horizontal_barriers=None)

    # win or loose money
    # money_making = true_value_fist_model * final_predictions

    # plotting
    plt.subplot(2, 1, 1)
    plt.plot(final_predictions.cumsum(), label="Final Prediction", color="black")
    plt.plot(true_value_fist_model.cumsum(), label = "Triple-barrier Labeld",color="goldenrod")
    # plt.plot(money_making[money_making > 0].cumsum(), label="Win Money", color="green")
    # plt.plot(money_making[money_making <= 0].cumsum(), label="Loose Money", color="red")
    plt.axvline(x=predictions_train.index[-1], color="navy", label="Train Data End")
    plt.ylabel("Predicted Values:")
    plt.xlabel("True Values:")
    plt.title(f"Cumulative Predictions: (side and size) vs. True values : (Trible barrier Labeld):")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.title(f"Distribution of the Cumulative Predicted Values vs. Cumulative True Values:")
    sns.distplot(final_predictions.cumsum(), fit=norm, label="Final Predictions")
    sns.distplot(true_value_fist_model.cumsum(), label="True Values", color="dimgray")
    plt.legend(loc="best")
    plt.show()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# the smaller the better (overfitting)
h = 0.005
# 0.01

# weekly the higher the better around 60 was best
number_days = 31

# false was always better
pca = False


# "a lower number is preffered as excessive partitioning would agian place in the testing set samples too similar to those ues in the testing set

# good
n_splits = 3

pct_emb = 0.1

number_cols = 5

compute_second_model(df,h,number_days,pca,number_cols,n_splits,pct_emb)





