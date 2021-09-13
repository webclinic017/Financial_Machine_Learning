"""
Name : validation_functions.py in Project: Financial_ML
Author : Simon Leiner
Date    : 18.05.2021
Description: Cross validation in finance (Classification)

Site of sklearn for differenet metrics:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, auc, classification_report, confusion_matrix, det_curve, plot_det_curve,roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from numpy import argmax
import warnings

# disable some warnings
warnings.filterwarnings(category=UserWarning,action="ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

class PurgedKFold(KFold):

    """Cross validation class inheriting KFold and meant to replace and enhance it"""

    def __init__(self, n_splits=5, pct_emb=0.01):
        """
        :param n_splits:  integer: number of splits
        :param pct_emb: float: embargo parameter
        """

        # Return a proxy object that delegates method calls to a parent or sibling class of type.
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.pct_emb = pct_emb

    # SEE: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_split.py

    # overwrite function from KFold class
    def split(self, X, y=None, groups=None):

        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # evenly spaced values between 0 and the number of rows of X with step size 1
        indices = np.arange(X.shape[0])

        # array: from 0 to number of rows in X eg: [0,...,386]
        # print(indices)

        # number of rows that are part of the embargo
        mbrg = int(X.shape[0] * self.pct_emb)

        # print(f"Number of rows excluded between train and testing set: {mbrg}")
        # print("-" * 10)

        # skalar: number of rows * pct eg: 3
        # print(mbrg)

        # split array indices into n_splits sub arrays
        data = np.array_split(indices, self.n_splits)

        # array of multiple arrays: split up n_splits times eg: [[0,...,77],[78,...,154],...]
        # print(data)

        # for subarrays in data: create tupel:(first and last element + 1
        test_starts = [(i[0], i[-1] + 1) for i in data]

        # array with tuples: eg: [(0, 78), (78, 155), (155, 232), (232, 309), (309, 386)]
        # print(test_starts)

        # loop trough tuple list
        for i, j in test_starts:

            # start of testing set
            start = X.index[i]

            # starting Date eg: 2019-12-02 00:00:00
            # print(start)

            # indices from i to j
            test_indices = indices[i:j]

            # array with indices: subsetted: eg [0,...,77]
            # print(test_indices)

            # get the max index
            max_index = X.index.searchsorted(X.index[test_indices].max())

            # scalar: eg:77
            # print(max_index)

            # indices from j + emb until the end
            train_indices = X.index.searchsorted(X[X.index < start].index)

            # array with indices: subsetted: eg [81,...,385]
            # print(train_indices)

            train_indices = np.concatenate((train_indices,indices[max_index+mbrg:]))

            # print(train_indices)

            # yield is a keyword that is used like return, except the function will return a generator.
            # Generators are iterators, a kind of iterable you can only iterate over once
            yield train_indices, test_indices

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def algorithmncomp(X_train, Y_train, cv,scoring):

    """

    This function creates an comparison of multiple basic classifiers.

    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :param cv: Type of cross validation object
    :param scoring: string: scoring method
    :return: best trained model
    """

    # create empty list for your models
    classifiers = []

    # estimators to use
    vote_est = [
        ('rf', RandomForestClassifier(criterion="entropy", class_weight="balanced_subsample")),
        ('svc', SVC(probability=True, class_weight="balanced")),
        ('xgb', XGBClassifier())
    ]

    # Voting build_model
    model_vote = VotingClassifier(estimators=vote_est, voting="soft")

    #append the models you want to compare to the list
    classifiers.append(SVC(probability=True, class_weight="balanced"))
    classifiers.append(RandomForestClassifier(criterion = "entropy", class_weight = "balanced_subsample"))
    classifiers.append(XGBClassifier())
    classifiers.append(GradientBoostingClassifier())
    classifiers.append(model_vote)

    #create an empty list in which we want to save our resulsts
    cv_results = []

    #for each build_model
    for classifier in classifiers:

        #compute a score with cross Validation
        score = cross_val_score(classifier, X_train, y=Y_train, scoring=scoring, cv=cv, n_jobs=10)

        #append the score to the result list
        cv_results.append(score)

    #create empty list for the mean and stcv of all different trained models, caused by cross validation
    #cross_val_score returns an Array of scores of the estimator for each run of the cross validation.
    cv_means = []
    cv_std = []

    #for each result
    for cv_result in cv_results:

        #append the mean
        cv_means.append(cv_result.mean())

        #append the std
        cv_std.append(cv_result.std())

    # create list with algorithm names
    algo_names = [
                  "SVC",
                  "RandomForestClassifier",
                  "XGBClassifier",
        "GradientBoostingClassifier",
                  "VotingClassifier"]

    #create a DF with the wanted Information
    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": algo_names})

    # get the best classifier:
    best_class_ = np.max(np.array(cv_res["CrossValMeans"]))

    # find the associated value
    best_classif_ = cv_res["Algorithm"][cv_res["CrossValMeans"] == best_class_]

    # get the first value
    best_classif_ = best_classif_.iloc[0]
    print(f"The best and thus chosen build_model is: {best_classif_} with a score of: {round(best_class_,4) * 100} %")
    print("-" * 10)

    # select the best build_model
    if best_classif_ == "SVC":
        model = classifiers[0]
    elif best_classif_ == "RandomForestClassifier":
        model = classifiers[1]
    elif best_classif_ == "XGBClassifier":
        model = classifiers[2]
    elif best_classif_ == "GradientBoostingClassifier":
        model = classifiers[3]
    elif best_classif_ == "VotingClassifier":
        model = classifiers[4]

    # figure size
    plt.figure(figsize=(15,8))

    # color palette
    cmap = sns.color_palette("rocket")

    #visualization by barplot
    sns.barplot(x ="CrossValMeans", y ="Algorithm", data=cv_res,palette=cmap)

    #stylistical facts
    plt.title(f"Machine Learning Algorithm {scoring} Score: \n")
    plt.xlabel(f"{scoring} score")
    plt.xlim(0,1)
    plt.ylabel("Algorithm")

    plt.show()

    return model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def score_by_cv(model, X_train, Y_train, cv,scoring):

    """

    This function computes a score with cross validation.

    :param model: trained build_model
    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :param cv: Type of cross validation object
    :param scoring: string: scoring method
    :return: float: mean of all differnet scoring cv paths

    Note for classification problems in finance we prefer the f1 as a good metric:
    F1 = 2 * (precision * recall) / (precision + recall) as it accounts for the tradeoff between precision and recall

    Alternatively one can only go for precision or recall but for more: SEE Confusion Matrix
    """

    # SEE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

    #get the score with our build_model
    n_scores = cross_val_score(model, X_train, Y_train, scoring=scoring, cv=cv, n_jobs= 10, error_score="raise")

    print(f"Mean {str(scoring)} score und Standarddeviation by cross validation: {round(n_scores.mean()*100,4)} %, ({round(n_scores.std()*100,4)} %)")
    print("-" * 10)

    return n_scores.mean()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def get_confusionmatrix(Y_true, y_pred, normalize = None):

    """

    This function plots the confusion matrix.

    :param Y_true: pd.Series: true y values
    :param y_pred: pd.Series: predictions py the build_model
    :param normalize: boolean: 'true' => normalizes the matrix
    :return: None

    Note:

    True : 1 : positive returns
    Negative : 0 : negative returns

    SEE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    #get the confusion matrix
    mat = confusion_matrix(Y_true, y_pred,normalize=normalize)

    # extract values:
    tn, fp, fn, tp = mat.ravel()
    pos = tp+fp
    neg = tn+fn
    total = tn + fp + fn + tp

    print(f"There are a Total of {total} training predictions:")
    print(f"The build_model predicted {pos} times ({round((pos / total), 2)}) % positive returns.")
    print(f"The build_model predicted {neg} times ({round((neg / total), 2)}) % negative returns.")
    print(f"The build_model predicted in total {round(((tp + tn) /total),2)} % of the training predictions correctly.")
    print("-" * 10)

    # plotting settings
    sns.set(font_scale = 1.25)

    #view it as heatmap:
    sns.heatmap(
        mat,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )

    #set stylisitcal stuff
    plt.title('Confusion Matrix: \n')
    plt.xlabel("Predicted label:")
    plt.ylabel("True label:")
    plt.show()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def classreport(Y_true, y_pred):

    """

    This function computes a classification report.

    :param Y_true: pd.Series: true y values
    :param y_pred: pd.Series: predictions py the build_model
    :return: None

    Note:

    True : 1 : positive returns
    Negative : 0 : negative returns

    TN / True Negative: the case was negative and predicted negative returns
    TP / True Positive: the case was positive and predicted positive returns
    FN / False Negative: the case was positive but predicted negative returns
    FP / False Positive: the case was negative but predicted positive returns

    Recall — What percent of the positive cases did you catch?

    Recall of negative returns: correctly classified negative returns from all true labels
    Recall of positive returns: correctly classified positive returns from all true labels

    Precision — What percent of your predictions were correct?

    Precision of negative returns: correctly classified negative returns from the predictions
    Precision of positive returns:

    The f1-score is the harmonic mean between precision & recall

    The support is the number of occurence of the given class in your dataset :  balanced dataset ?

    # SEE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """

    # class names : 0, then 1: ordern increasingly :
    # class_names = ['class 0', 'class 1', 'class 2']

    class_names = ['negative returns:', 'positive returns:']

    print("Classification Report:")
    print(classification_report(Y_true, y_pred, target_names=class_names))
    print("-" * 10)

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_roc_det_pr_curve_test(model, X_test, Y_test):

    """

    This function plots the ROC Det and precision recall Curve.

    :param model: trained build_model
    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :return: None

    # See:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html
    https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-roc-curve-visualization-api-py

    """

    # create figure, axes object
    fig, [ax_roc, ax_det, ax_pr] = plt.subplots(1, 3, figsize=(11, 5))

    # plot the roc curve
    plot_roc_curve(model, X_test, Y_test, ax=ax_roc)

    # plot the det curve
    plot_det_curve(model, X_test, Y_test, ax=ax_det)

    # plot the pr curve
    plot_precision_recall_curve(model, X_test, Y_test, pos_label=1, ax=ax_pr)

    # plot a diagonal line through the figure of the roc cruve
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Min Line', alpha=.8)

    # set the xlim and ylim and give the chart a title
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="Receiver Operating Characteristic (ROC) curve [Test]:")
    ax_det.set(title='Detection Error Tradeoff (DET) curve [Test]:')
    ax_pr.set(title='Precision Recall curve [Test]:')

    # add a legend
    ax_roc.legend(loc="best")
    ax_det.legend(loc="best")
    ax_pr.legend(loc="best")

    plt.show()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_roc_det_pr_curve_cv(model,cv,X_train,y_train):

    """

    This function plots the ROC, DET and PR Curve with cross validation.

    :param model: trained build_model
    :param cv: Type of cross validation object
    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :return: float: best threshold value

    See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    """

    # create empty lists for saving data

    # true positive rate
    tprs = []

    # auc scores
    aucs = []

    # best thresholds
    best_threshs = []

    # false positive rate
    mean_fpr = np.linspace(0, 1, 100)

    # create figure, axes object
    fig, [ax_roc, ax_det, ax_pr] = plt.subplots(1, 3, figsize=(11, 5))

    # use overwritten split function to split the Dataframe into different subsets
    for i, (train, test) in enumerate(cv.split(X_train, y_train)):

        # fit the build_model
        model.fit(X_train.iloc[train], y_train.iloc[train])

        # See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

        # plot the roc curve
        viz_roc = plot_roc_curve(model, X_train.iloc[test], y_train.iloc[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax_roc)

        # plot the det curve
        plot_det_curve(model, X_train.iloc[test], y_train.iloc[test],
                             name='DET fold {}'.format(i),
                             alpha=0.8, lw=1, ax=ax_det)

        # plot the pr curve
        plot_precision_recall_curve(model, X_train.iloc[test], y_train.iloc[test],
                                             name='PR fold {}'.format(i),ax=ax_pr,pos_label=1)

        # linear interpolate values
        interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)

        # set starting value to 0
        interp_tpr[0] = 0.0

        # get the best threshold yb the youden's J statistic
        t_stat = viz_roc.tpr - viz_roc.fpr

        # find the appropriate index
        index_ = argmax(t_stat)

        # get the best threshold
        best_thresh = viz_roc.thresholds[index_]

        # append the results to the list
        tprs.append(interp_tpr)
        aucs.append(viz_roc.roc_auc)
        best_threshs.append(best_thresh)

    # plot a diagonal line through the figure
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Min Line', alpha=.8)

    # get the mean
    mean_tpr = np.mean(tprs, axis=0)

    # set the last entry as 1
    mean_tpr[-1] = 1.0

    # See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc

    # get the area under the curve
    mean_auc = auc(mean_fpr, mean_tpr)

    # get the sd of the AUc scores
    std_auc = np.std(aucs)

    # plot the means of both rates
    ax_roc.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)

    # get the maximum value, default 1
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    # get the minimal value, default 0
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # fil between lines
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # set the xlim and ylim and give the chart a title
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="Receiver Operating Characteristic (ROC) curves [CV]:")
    ax_det.set(title='Detection Error Tradeoff (DET) curves [CV]:')
    ax_pr.set(title='Precision Recall curves [CV]:')

    # add a legend
    ax_roc.legend(loc="best")
    ax_det.legend(loc="best")
    ax_pr.legend(loc="best")

    # get the total best best thresholds
    best_thresh_total = float(np.mean(best_threshs, axis=0))

    print(f"The best threshold value by CV is: {round(best_thresh_total,2)}.")
    print("-" * 10)

    plt.show()

    return best_thresh_total

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_learning_curve_cv(model, X_train, Y_train,cv):

    """

    This function generates a plot with the learning curve of the Training and CV - Score.

    :param model: trained build_model
    :param X_train: pd.DataFrame: Full X training set
    :param Y_train: pd.Series: Fully y training set
    :param cv: Type of cross validation object
    :return: None

    Learning curve:

    Train Learning Curve: Learning curve calculated from the training dataset that gives an idea of how well the
    build_model is learning.

    Validation Learning Curve: Learning curve calculated from a hold-out validation dataset
    that gives an idea of how well the build_model is generalizing.
    """

    # stylisitcal facts
    plt.figure()
    plt.title(f"Learning curve [CV]: \n")
    plt.xlabel("Number of Observations:")
    plt.ylabel(f"Neg log Loss Score:")
    plt.xlim(0, int(len(Y_train)))

    # create the learning cure with your given estimator
    # train_size = number of training examples that has benn used to generate the learning curve
    # train_scores = Scores of the training dataset
    # test_scores = scores on the test set

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, Y_train, cv=cv,scoring="neg_log_loss",train_sizes=np.linspace(.1, 1.0, 5))

    # get the mean and std for plotting
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # create a plot grid for multpiple plots in one figure
    plt.grid()

    # Fill the area between two horizontal curves.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="darkred")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="lightcoral")

    # plot
    plt.plot(train_sizes, train_scores_mean, 'o-', color="darkred",
             label="Training Learning Curve")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="lightcoral",
             label="Cross-validation Learning Curve")

    # add a legend
    plt.legend(loc="best")

    plt.show()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
