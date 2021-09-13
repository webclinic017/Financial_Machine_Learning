"""
Name : parameter_tuning_functions.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description:
"""

from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings

# don't show any warnings and printing options
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def get_hyper_tuning(model,x_train,y_train, cv = None,scoring = "f1"):

    """

    This function computes the grid search Algorithm for different models.

    :param model: build_model to hypertune
    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :param cv: Type of cross validation object
    :param scoring: string: scoring method
    :return: best parameters and estimators
    """

    # get the build_model name
    modelname = type(model).__name__

    # select a Parameter Grid for your build_model:
    if modelname == "RandomForestClassifier":
        param_grid = {
            "n_estimators": [100,1000,1200],
            "max_depth": [5,20,None],
            "min_samples_split": [0.1,0.9,10],
            "min_samples_leaf": [0.1,0.5, 1],
            "min_weight_fraction_leaf": [0, 0.05],
            "max_features": ["auto","log2"],
        }

    elif modelname == "XGBClassifier":

        param_grid = {
            "n_estimators": [100,1000,1200],
            "early_stopping_rounds": [5, 10],
            "learning_rate": [0.001,0.02,0.05],
            'max_depth': [8, 10, 15],
            'gamma': [0.1, 0.3, 0.6],
            'min_child_weight': [1,3,8],
            'subsample': [0.001, 0.005, 0.01, 0.02],
            'colsample_bytree': [0.3,0.5,10.8],

        }

    elif modelname == "SVC":

        param_grid = {
            "C": [1e-2,1,10,50,100],
            "kernel": ["rbf", "sigmoid"],
            "gamma": [1e-2,0.1, "scale", "auto"],
            "tol": [1e-2, 1e-3, 1e-4],
        }

    elif modelname == "GradientBoostingClassifier":

        param_grid = {
            "learning_rate": [0.001,0.02,0.05],
            "n_estimators": [100,1000,1200],
            'max_depth': [8, 10, 15],
            "min_samples_split": [0.1,0.9,10],
            "min_samples_leaf": [0.1, 0.5, 1],
        }

    elif modelname == "VotingClassifier":
        param_grid = {
            # "learning_rate": [0.001,0.02,0.05],
            # "n_estimators": [100,1000,1200],
            # 'max_depth': [8, 10, 12, 15],
            # "min_samples_split": [0.1, 0.9, 10],
            # "min_samples_leaf": [0.1, 0.5, 1],
            # "max_features": ["auto","log2"],
        }

    else:
        print(f"Couldn't tune the Hyper parameters as the Model with name {modelname} couldn't match.")
        return model

    #gridsearch uses cross validation
    gs = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring,n_jobs=10)

    #fit your build_model with each different kombination in the grid
    gs.fit(x_train, y_train)

    # get the best parameters and estimators
    best = gs.best_estimator_

    print(f"Best Estimators: {best}")

    print("-" * 10)

    return best

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
