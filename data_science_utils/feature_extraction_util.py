import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt


def relax_data(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str):
    """
    Performs data relaxation on a common column of test and training dataframe.
    With data relaxation all values that appear 3 or more times more often in the trainings dataframe than in
    the test dataframe are removed. This prevents inaccurate predictions with unknown or almost unknown values of
    the provided feature column.
    :param df_train: trainings dataframe
    :param df_test: test dataframe
    :param col: column where data relaxation should be applied. This column needs to be present in both dataframes.
    :return: training and test dataframe with relaxed column
    """
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col: 'train'}, axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col: 'test'}, axis=1))
    cv3 = pd.merge(cv1, cv2, on='index', how='outer')
    factor = len(df_test) / len(df_train)
    cv3['train'].fillna(0, inplace=True)
    cv3['test'].fillna(0, inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train) / 10000)
    cv3['remove'] = cv3['remove'] | (factor * cv3['train'] < cv3['test'] / 3)
    cv3['remove'] = cv3['remove'] | (factor * cv3['train'] > 3 * cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove'] == False else 0, axis=1)
    cv3['new'], _ = cv3['new'].factorize(sort=True)
    cv3.set_index('index', inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)
    return df_train, df_test


def covariate_shift(train: pd.DataFrame, test: pd.DataFrame, params: dict, column: str):
    """
    Calculates a roc auc score of a given column which indicates if the feautes has a covariate shift.
    In order to calculate the score a simple gradient boosting classifier is trained on that feature.
    AUC-Scores near 0.5 indicate no covariate shift. Everything above ~0.7 indicates a covariate shift in that feature.
    In this case data relaxation or other methods to remove the disparity between test and train dataset should be
    applied.
    :param train: train dataframe
    :param test: test dataframe
    :param params: parameters for the lightgbm-tree
    :param column: feature column which must be present in both dataframes
    :return: auc score indicating if the feature has a covariate shift
    """
    df_card1_train = pd.DataFrame(data={column: train[column], 'isTest': 0})
    df_card1_test = pd.DataFrame(data={column: test[column], 'isTest': 1})

    # Creating a single dataframe
    df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)

    # Encoding if feature is categorical
    if str(df[column].dtype) in ['object', 'category']:
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[column], df['isTest'], test_size=0.33,
                                                        stratify=df['isTest'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])

    del df, X_train, y_train, X_test, y_test

    return roc_auc


def correct_features_with_covariate_shift(train, test, params, columns, threshold=0.65):
    """
    Automatically corrects feature column with a covariate shift (AUC-Score>=threshold) by performing data relaxation.
    :param train: training dataframe containing the columns that should be checked/corrected
    :param test: test dataframe the columns that should be checked/corrected
    :param params: parameters for the lightgbm-tree
    :param columns: columns that should be checked/corrected
    :param threshold: the threshold when data relaxation should be performed (AUC-Score>=threshold)
    :return: a modified version of the train and test dataframe
    """
    train = train.copy()
    test = test.copy()
    for column in tqdm(columns):
        auc_score = covariate_shift(train, test, params, column)
        if auc_score >= threshold:
            print(column)
            train, test = relax_data(train, test, column)
            new_score = covariate_shift(train, test, params, column)
            print("Performing data relaxation on column {} (New AUC-Score:{} Old Score: {})".format(column, new_score,
                                                                                                    auc_score))

    return train, test


def evaluate_features(X_train, y_train, X_test, y_test, params, metrices):
    """
    Trains a simple gradient boosting model and evaluates its feature importances (if multiple columns provided).
    Furthermore the trained model is evaluated with the provided metric(es).
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param params:
    :param metrices:
    :return:
    """

    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
        X_train[col] = le.transform(list(X_train[col].astype(str).values))
        X_test[col] = le.transform(list(X_test[col].astype(str).values))


    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train.values, y_train)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    features_to_show = len(X_train.columns)

    plt.figure(figsize=(15,10))
    plt.title("Feature importances")
    plt.bar(range(features_to_show), importances[indices][:features_to_show],
            color="r", align="center")
    feature_names = [X_train.columns[indices[f]] for f in range(features_to_show)]
    plt.xticks(range(features_to_show), feature_names, rotation='vertical')
    plt.xlim([-1, features_to_show])
    plt.show()

    scores = get_model_scores(clf, X_train, y_train, X_test, y_test, metrices, True)

    df_feature_importance = pd.DataFrame({'column':X_train.columns[indices], 'importance':importances[indices]})
    return (df_feature_importance, scores)

def get_model_scores(model, x_train, y_train, x_test, y_test, metrices, print_values=True):
    scores = {}
    for metric in metrices:
        try:
            score_train = metric(model.predict(x_train), y_train)
            score_test = metric(model.predict(x_test), y_test)
            if print_values:
                print(metric.__name__, "(train):", score_train)
                print(metric.__name__, "(test):", score_test)
                print("------------------------------------------------------------")
            scores[metric.__name__] = [score_train, score_test]
        except:
            print("Could not calculate score", metric.__name__)
            print("------------------------------------------------------------")
            scores[metric.__name__] = [None, None]
    return scores
