"""
Name : model_interpretation.py in Project: Financial_ML
Author : Simon Leiner
Date    : 02.09.2021
Description: Interpret the build_model with nice plots
"""

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from build_model import model_building, model_insights
import positioning
import warnings

# disable some warnings
warnings.filterwarnings(category=FutureWarning,action="ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def interprate_model(model, y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test):

    """

    This function interprates the model given the binary +-1 Information.


    :param model: trained build_model
    :param y_pred_cv: pd.Series: CV predictions: +-1
    :param y_pred_test: pd.Series: Test predictions: +-1
    :param y_prob_cv: pd.Series: CV predictions of probabilities
    :param y_prob_test: pd.Series: Test predictions of probabilities
    :param x_train:pd.DataFrame: X values of the training data
    :param x_test:pd.DataFrame: X values of the testing data
    :param y_train: pd.Series: y values of the training data
    :param y_test: pd.Series: y values of the testing data
    :return: None
    """

    # define the strategy
    strategy = "buy"

    # append the training and testing predictions, true values and predicted probabilities
    predictions, true_values, predictions_prob,train_end_date = model_building.compute_predictions_true_values(y_pred_cv,y_pred_test,y_prob_cv,y_prob_test,x_train, x_test, y_train, y_test)
    # print(predictions)

    # class distribution of the predictions
    distrib_pred = predictions.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_pred)

    # class distribution of the true values
    distrib_true = true_values.value_counts(normalize=True).to_frame().sort_index(ascending=True)
    # print(distrib_true)

    # plotting
    plt.subplot(2, 1, 1)
    plt.plot(predictions.cumsum(),label = "Prediction", color="black")
    plt.plot(true_values.cumsum(), label = "True Value",color="goldenrod")
    plt.axvline(x=train_end_date,color = "navy",label = "Train Data End")
    plt.ylabel("Predicted & True Values +1 or -1:")
    plt.title(f"Cumulative Predictions vs. True values:")
    plt.legend(loc = "best")

    # plotting
    plt.subplot(2, 1, 2)
    plt.title(f"Distribution of the Cumulative Predicted Values vs. True Values:")
    sns.distplot(predictions.cumsum(),label="Predictions")
    sns.distplot(true_values.cumsum(),label="True Values",color="dimgray",fit = norm)
    plt.xlabel("Predicted & True Values +1 or -1:")
    plt.legend(loc="best")
    plt.show()

    # calculate the "money making process": just side, just binary data
    money_making_binary = positioning.choose_strategy_binary(predictions,true_values,strategy)

    # plotting
    plt.subplot(2, 1, 1)
    plt.plot(money_making_binary.cumsum(), label="Correct Classified", color="black")
    plt.axvline(x=train_end_date, color="navy", label="Train Data End")
    plt.ylabel("Predicted & True Values +1, -1 or 0:")
    plt.title(f"Cumulative Exploitation of the Correct Side prediction with strategy {strategy}:")
    plt.legend(loc="best")

    # plotting
    plt.subplot(2, 1, 2)
    plt.title(f"Distribution of the Cumulative correct Side classification with strategy {strategy}:")
    sns.distplot(money_making_binary.cumsum(),fit = norm)
    plt.xlabel("Predicted & True Values +1, -1 or 0:")
    plt.show()

    print(
        "Class distribution of the predictions vs the true values (training and testing datasets) by the first build_model:")
    print(
        f"Positive Returns  1 : Predicted: {round(distrib_pred.iloc[1, 0], 4) * 100} % vs. True {round(distrib_true.iloc[1, 0], 4) * 100} %")
    print(
        f"Negative Returns -1 : Predicted: {round(distrib_pred.iloc[0, 0], 4) * 100} % vs. True {round(distrib_true.iloc[0, 0], 4) * 100} %")
    print("-" * 10)

    # get build_model insights by permutation importance
    model_insights.perm_importance(model, x_test, y_test)

    return None