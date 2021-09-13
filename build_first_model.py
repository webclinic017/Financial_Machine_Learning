"""
Name : build_first_model.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: build the first build_model
"""

import pandas as pd
from adjust_data import adjust_dataframe
from select_feature import feature_selection
import validation_functions
import parameter_tuning_functions
from sklearn.model_selection import cross_val_predict
import bet_sizing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def compute_first_model(df,factors,h ,number_days ,pca,number_cols,n_splits, pct_emb,side = None):

    """ This function computes the first build_model


    # See: https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e
    """

    # get the finished Dataframes
    data_cleaner,df_previous = adjust_dataframe.adjust_data(df, factors, h, number_days, pca,side)

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
    x_train = df_train.drop(["t+1"], axis=1)

    y_train = df_train["t+1"]

    # divide the test Dataframe in X and Y
    x_test = df_test.drop(["t+1"], axis=1)

    y_test = df_test["t+1"]

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
    y_predprob_cv = cross_val_predict(best_model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=10)

    # plot the roc curve and get the auc score: get the best threshold lvl
    threshold = validation_functions.plot_roc_curve_cv(best_model, cv, x_train, y_train)

    # adjusted predictions
    y_pred_adjusted_cv = bet_sizing.adjust_predictions_by_threshold(y_predprob_cv, threshold)

    # get the confusion matrix, showing true and predicted values with cross validation
    validation_functions.get_confusionmatrix(y_train, y_pred_adjusted_cv)

    # evaluation by classreport with cross validation
    validation_functions.classreport(y_train, y_pred_adjusted_cv)

    # plot the learning curve
    validation_functions.plot_learning_curve(best_model, x_train, y_train, cv,scoring)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # predictions with final test set
    y_pred_test = best_model.predict(x_test)

    # predictions with final test set
    y_predprob_test = best_model.predict_proba(x_test)

    # plot the roc curve and get the auc score
    # validation_functions.plot_roc_curve_test(best_model, x_test, y_test)

    # get the confusion matrix, showing true and predicted values with the test set
    # validation_functions.get_confusionmatrix(y_test, y_pred_test)

    # evaluation by classreport with the test set
    # validation_functions.classreport(y_test, y_pred_test)

    # convert the final predictions to a pandas Series and append the prices
    predictions_train = pd.Series(y_pred_adjusted_cv, index=x_train.index)
    predictions_test = pd.Series(y_pred_test, index=x_test.index)

    # append the predictions together
    predictions = predictions_train.append(predictions_test)
    # print(predictions)

    # true values
    true_value_fist_model = y_train.append(y_test)
    # print(true_value_fist_model)

    # class distribution of the predictions
    distrib_pred = predictions.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_pred)

    # class distribution of the predictions
    distrib_true = true_value_fist_model.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_true)

    print("Class distribution of the predictions vs the true values (training and testing datasets) by the first build_model:")
    print(f"Positive Returns  1 : Predicted: {round(distrib_pred.iloc[1, 0], 4) * 100} % vs. True {round(distrib_true.iloc[1, 0], 4) * 100} %")
    print(f"Negative Returns -1 : Predicted: {round(distrib_pred.iloc[0, 0], 4) * 100} % vs. True {round(distrib_true.iloc[0, 0], 4) * 100} %")
    print("-" * 10)

    # plotting
    plt.subplot(2, 1, 1)
    plt.plot(predictions.cumsum(),label = "Prediction", color="black")
    plt.plot(true_value_fist_model.cumsum(), label = "True Value",color="goldenrod")
    plt.axvline(x=predictions_train.index[-1],color = "navy",label = "Train Data End")
    plt.ylabel("Predicted Values +1 or -1:")
    # plt.xlabel("True Values +1 or -1:")
    plt.title(f"Cumulative Predictions vs. True values of the one step ahead triple-barrier-label data with a span of {number_days} days:")
    plt.legend(loc = "best")

    plt.subplot(2, 1, 2)
    plt.title(f"Distribution of the Cumulative Predicted Values vs. Cumulative True Values:")
    sns.distplot(predictions.cumsum(), fit=norm,label="Predictions")
    sns.distplot(true_value_fist_model.cumsum(),label="True Values",color="dimgray")
    plt.legend(loc="best")
    plt.show()

    return cv, predictions, true_value_fist_model, df_previous, wanted_columns

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



