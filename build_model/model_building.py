"""
Name : model_building.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: only the first build_model as a whole
"""

from adjust_data import adjust_dataframe
from select_feature import feature_selection
from validate_model import validation_functions
from sklearn.model_selection import cross_val_predict
import bet_sizing_first_model
import numpy as np
import pandas as pd

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def compute_first_model(ps,factors,h ,numdays,days_predict_into_future ,pca,number_cols,n_splits, pct_emb,cumsum):

    """

    This function trains the first model.

    :param ps: pd.Series: Prices
    :param factors: float: that sets the size of the 2 horizontal barriers
    :param h: float: filter size for the CUMSUM Filter
    :param numdays: integer: number of days to add for vertical barrier
    :param days_predict_into_future: integer: number of days to predict in the future
    :param pca: boolean: apply pca transfomration
    :param number_cols: integer: fraction of columns to select
    :param n_splits: integer: number of splits for cross validations
    :param pct_emb: float: percentage of points to exclude for cross validation
    :param cumsum: boolean: wether to apply the cumsum filter or not
    :return: pd.Series: true values, pd.Series: prediction, trained model
    """

    # get the finished Dataframes
    data_cleaner, df_label = adjust_dataframe.adjust_data(ps, factors, h, numdays,days_predict_into_future, pca, cumsum)

    # training set
    df_train = data_cleaner[0]

    # testing set
    df_test = data_cleaner[1]

    # get the wanted columns
    wanted_columns = feature_selection.select_wanted_cols(df_train,number_cols=number_cols)

    # subset X by selected columns
    df_train = df_train[wanted_columns]
    df_test = df_test[wanted_columns]

    # divide the train Dataframe in X and Y
    x_train = df_train.drop(["t"], axis=1)
    y_train = df_train["t"]

    # divide the test Dataframe in X and Y
    x_test = df_test.drop(["t"], axis=1)
    y_test = df_test["t"]

    # get cross validation object
    cv = validation_functions.PurgedKFold(n_splits=n_splits, pct_emb=pct_emb)

    # scoring function: in the first build_model, we want to achieve a high recall / precision, f1
    scoring = "f1"

    # compare the models
    best_model = validation_functions.algorithmncomp(x_train, y_train, cv, scoring)

    # tune the hyper parameters of the best build_model
    # best_model = parameter_tuning_functions.get_hyper_tuning(best_model, x_train, y_train, cv, scoring)

    print("Finished with Hyperparameter Tuning of the first build_model.")
    print("-" * 10)

    # compare score by scoring function with cross validation
    validation_functions.score_by_cv(best_model, x_train, y_train, cv, scoring)

    # make predictions with cross validation
    y_prob_cv = cross_val_predict(best_model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=10)

    # plot the roc curve and get the auc score: get the best threshold lvl
    threshold = validation_functions.plot_roc_det_pr_curve_cv(best_model, cv, x_train, y_train)

    # adjusted predictions
    y_pred_cv = bet_sizing_first_model.adjust_predictions_by_threshold(y_prob_cv, threshold)

    # get the confusion matrix, showing true and predicted values with cross validation
    validation_functions.get_confusionmatrix(y_train, y_pred_cv)

    # evaluation by classreport with cross validation
    validation_functions.classreport(y_train, y_pred_cv)

    # plot the learning curve
    validation_functions.plot_learning_curve_cv(best_model, x_train, y_train, cv)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # predictions with final test set
    y_pred_test = best_model.predict(x_test)

    # predictions with final test set
    y_prob_test = best_model.predict_proba(x_test)

    # plot the roc curve and get the auc score
    validation_functions.plot_roc_det_pr_curve_test(best_model, x_test, y_test)

    # get the confusion matrix, showing true and predicted values with the test set
    validation_functions.get_confusionmatrix(y_test, y_pred_test)

    # evaluation by classreport with the test set
    validation_functions.classreport(y_test, y_pred_test)

    return best_model, y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test, df_label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def compute_predictions_true_values(y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test):

    """

    This function compute the predictions, the probabilities thereof and the true values of the training and testing set.

    :param y_pred_cv: pd.Series: CV predictions: +-1
    :param y_pred_test: pd.Series: Test predictions: +-1
    :param y_prob_cv: pd.Series: CV predictions of probabilities
    :param y_prob_test: pd.Series: Test predictions of probabilities
    :param x_train:pd.DataFrame: X values of the training data
    :param x_test:pd.DataFrame: X values of the testing data
    :param y_train: pd.Series: y values of the training data
    :param y_test: pd.Series: y values of the testing data
    :return: pd.Series: predictions, pd.Series: true_values, pd.Series: predictions_prob, datetime: train_end_date
    """

    # convert the final predictions to a pandas Series and append the prices
    predictions_train = pd.Series(y_pred_cv, index=x_train.index)
    predictions_test = pd.Series(y_pred_test, index=x_test.index)

    # append the predictions together
    predictions = predictions_train.append(predictions_test)
    # print(predictions)

    # true values
    true_values = y_train.append(y_test)
    # print(true_value_fist_model)

    # convert the values to a numpy array
    y_predprob_cv = np.array(y_prob_cv)
    y_predprob_test = np.array(y_prob_test)
    # print(y_predprob_cv)

    # select the probability from the class x elemt -1,1 : select class 1: second column
    correct_confidences_cv = y_predprob_cv[:, 1]
    correct_confidences_test = y_predprob_test[:, 1]
    # print(correct_confidences_cv)

    # convert the values to a pandas Series
    predictions_train_prob = pd.Series(correct_confidences_cv, index=x_train.index)
    predictions_test_prob = pd.Series(correct_confidences_test, index=x_test.index)

    # append the predictions together
    predictions_prob = predictions_train_prob.append(predictions_test_prob)
    # print(predictions_prob)

    # training data end
    train_end_date = predictions_train.index[-1]

    return predictions, true_values, predictions_prob, train_end_date

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #