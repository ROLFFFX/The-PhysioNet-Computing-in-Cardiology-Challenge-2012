# HW3 - helper.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import yaml
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

import hw3_main
import hw3_challenge


def get_train_test_split():
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population

    After all these steps, it splits the data into 80% train and 20% test.

    The binary labels take two values:
        -1: survivor
        +1: died in hospital

    Returns the features and labesl for train and test sets, followed by the names of features.
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels[:2500]
    IDs = df_labels['RecordID'][:2500]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))

    features = Parallel(n_jobs=16)(delayed(hw3_main.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values
    X = hw3_main.impute_missing_values(X)
    X = hw3_main.normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    return X_train, y_train, X_test, y_test, feature_names


def get_classifier(penalty='l2', C=1.0, class_weight=None):
    """
    Returns a logistic regression classifier based on the given
    penalty function, regularization parameter C, and class weights.
    """
    if penalty == 'l2':
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver='lbfgs', max_iter=1000)
    elif penalty == 'l1':
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver='saga', max_iter=5000)
    else:
        raise ValueError('Error: unsupported configuration')


def get_challenge_data():
    """
    This function is similar to helper.get_train_test_split, except that:
    - It reads in all 10,000 training examples
    - It does not return labels for the 2,000 examples in the heldout test set
    You should replace your preprocessing functions (generate_feature_vector,
    impute_missing_values, normalize_feature_matrix) with updated versions for the challenge
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))

    features = Parallel(n_jobs=16)(delayed(hw3_challenge.generate_feature_vector_challenge)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = hw3_challenge.impute_missing_values_challenge(X)
    X = hw3_challenge.normalize_feature_matrix_challenge(X)
    return X[:10000], y[:10000], X[10000:], feature_names


def make_challenge_submission(y_label, y_score):
    """
    Takes in `y_label` and `y_score`, which are two list-like objects that contain
    both the binary predictions and raw scores from your classifier.
    Outputs the prediction to challenge.csv.

    Please make sure that you do not change the order of the test examples in the heldout set
    since we will use this file to evaluate your classifier.
    """
    print('Saving challenge output...')
    pd.DataFrame({'label': y_label.astype(int), 'risk_score': y_score}).to_csv('challenge.csv', index=False)
    print('challenge.csv saved')
    return


def test_challenge_output():
    import csv
    with open('challenge.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile)
        i = 0
        for row in filereader:
            if i == 0:
                if row[0] != 'label':
                    print('INVALID COLUMN NAME: column name is not label.')
            else:
                rating = int(row[0])
                if rating != -1 and rating != 0 and rating != 1:
                    print('INVALID VALUE: values need to be -1, 0, or 1.')
            i += 1
        if i != 2001:
            print('INVALID NUMBER OF ROWS: number of rows is not 2001.')
        print('SUCCESS: csv file is valid.')
    return

