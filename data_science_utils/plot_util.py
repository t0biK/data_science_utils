import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_numerical(train: pd.DataFrame, test: pd.DataFrame, feature: str, target: str):
    """
    Plot some information about a numerical feature for both train and test set.
    Target must be a binary classification value (0 and 1).
    :param train (pandas.DataFrame): training set
    :param test (pandas.DataFrame): testing set
    :param feature (str): name of the feature
    :param target (str): name of the target feature
    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
    sns.kdeplot(train[feature], ax=axes[0][0], label='Train');
    sns.kdeplot(test[feature], ax=axes[0][0], label='Test');

    sns.kdeplot(train[train[target] == 0][feature], ax=axes[0][1], label='isFraud 0')
    sns.kdeplot(train[train[target] == 1][feature], ax=axes[0][1], label='isFraud 1')

    test[feature].index += len(train)
    axes[1][0].plot(train[feature], '.', label='Train');
    axes[1][0].plot(test[feature], '.', label='Test');
    axes[1][0].set_xlabel('row index');
    axes[1][0].legend()
    test[feature].index -= len(train)

    axes[1][1].plot(train[train[target] == 0][feature], '.', label='isFraud 0');
    axes[1][1].plot(train[train[target] == 1][feature], '.', label='isFraud 1');
    axes[1][1].set_xlabel('row index');
    axes[1][1].legend()

    pd.DataFrame({'train': [train[feature].isnull().sum()], 'test': [test[feature].isnull().sum()]}).plot(kind='bar',
                                                                                                          rot=0,
                                                                                                          ax=axes[2]
                                                                                                          [0]);
    pd.DataFrame({'isFraud 0': [train[(train[target] == 0) & (train[feature].isnull())][feature].shape[0]],
                  'isFraud 1': [train[(train[target] == 1) & (train[feature].isnull())][feature].shape[0]]}).plot \
        (kind='bar', rot=0, ax=axes[2][1]);

    fig.suptitle(feature, fontsize=18);
    axes[0][0].set_title('Train/Test KDE distribution');
    axes[0][1].set_title('Target value KDE distribution');
    axes[1][0].set_title('Index versus value: Train/Test distribution');
    axes[1][1].set_title('Index versus value: Target distribution');
    axes[2][0].set_title('Number of NaNs');
    axes[2][1].set_title('Target value distribution among NaN values');


def plot_categorical(train: pd.DataFrame, test: pd.DataFrame, feature: str, target: str, values: int = 5):
    """
    Plotting distribution for the selected amount of most frequent values between train and test
    along with distibution of target
    :param train (pandas.DataFrame): training set
    :param test (pandas.DataFrame): testing set
    :param feature (str): name of the feature
    :param target (str): name of the target feature
    :param values (int): amount of most frequest values to look at
    """
    df_train = pd.DataFrame(data={feature: train[feature], 'isTest': 0})
    df_test = pd.DataFrame(data={feature: test[feature], 'isTest': 1})
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[df[feature].isin(df[feature].value_counts(dropna=False).head(values).index)]
    train = train[train[feature].isin(train[feature].value_counts(dropna=False).head(values).index)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    sns.countplot(data=df.fillna('NaN'), x=feature, hue='isTest', ax=axes[0]);
    sns.countplot(data=train[[feature, target]].fillna('NaN'), x=feature, hue=target, ax=axes[1]);
    axes[0].set_title('Train / Test distibution of {} most frequent values'.format(values));
    axes[1].set_title('Train distibution by {} of {} most frequent values'.format(target, values));
    axes[0].legend(['Train', 'Test']);


def heatMap(df, mirror):
    # Create Correlation df
    corr = df.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    if mirror == True:
        # Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
        # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
        # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
        # show plot

    else:
        # Drop self-correlations
        dropSelf = np.zeros_like(corr)
        dropSelf[np.triu_indices_from(dropSelf)] = True
        # Generate Color Map
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        # Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
        # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
        # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
    # show plot
    plt.show()
