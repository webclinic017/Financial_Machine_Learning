"""
Name : model_insights.py in Project: Financial_ML
Author : Simon Leiner
Date    : 02.09.2021
Description: Get build_model insights by permutation importance
"""

import eli5
from eli5.sklearn import PermutationImportance

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def perm_importance(trained_model, X_test,Y_test):

    """

    This function shows, what features have the most impact.

    :param trained_model: your final trained build_model
    :param X_test: pd.DataFrame: Full X testing set
    :param Y_test: pd.Series: Fully y testing set
    :return: None

    Permutation feature importance is a build_model inspection technique that can be used for any fitted estimator
    when the data is tabular. This is especially useful for non-linear or opaque estimators. The permutation
    feature importance is defined to be the decrease in a build_model score when a single feature value is randomly
    shuffled 1. This procedure breaks the relationship between the feature and the target, thus the drop in the
    build_model score is indicative of how much the build_model depends on the feature. This technique benefits from being build_model
    agnostic and can be calculated many times with different permutations of the feature.
    """

    #make use of Permutation
    perm = PermutationImportance(trained_model,random_state = 1)
    # n_repeats int, default = 10

    # fit the build_model
    perm.fit(X_test, Y_test)

    #get the resulst as a DF: values to the top are the most important features
    df_weights = eli5.explain_weights_df(perm)

    #adjust feature names
    df_weights["feature"] = X_test.columns.tolist()

    print("Permutation Importance of the Variables [Test]:")
    print(df_weights.head(5))
    print("-" * 10)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
