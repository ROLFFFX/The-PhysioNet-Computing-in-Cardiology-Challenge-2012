# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
    # im suppressing this warning becasue in generate_feature_vector 
    # there will be all-nan value slice errors when calculating the 
    # median (most likely what happened is the patient passed away in 
    # the first sub period).  considering that im imputing these values 
    # in the next step, i would just ignore it for now.


import evaluation
from helper import *

def generate_feature_vector_challenge(df):
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}
    sub_periods = {
        "first_24_hours": {var: [] for var in timeseries_variables},
        "second_24_hours": {var: [] for var in timeseries_variables}
    }
    for index, rows in df.iterrows():
        variable_name = rows['Variable']
        value = rows['Value']
        time = int(rows['Time'].split(':')[0])
        # static variables
        if variable_name in static_variables:
            if value == -1:
                value = np.nan
            feature_dict[variable_name] = value
            
        # time-series variables
        elif variable_name in timeseries_variables:
            if value == -1:
                value = np.nan
            if time < 24:
                sub_periods['first_24_hours'][variable_name].append(value)
            else:
                sub_periods['second_24_hours'][variable_name].append(value)
        
    # calculate median instead of mean
    for period_name, period_data in sub_periods.items():
        for var, values in period_data.items():
            if values:
                feature_dict[f"{period_name}_median_{var}"] = np.nanmedian(values)
            else:
                feature_dict[f"{period_name}_median_{var}"] = np.nan

    return feature_dict

def impute_missing_values_challenge(X):
    nan_indices = set()
    # first scan: find indices with nan
    for x in X:
        # for each patient's record
        for i in range(0, len(x)):
            val = x[i]
            if (np.isnan(val)):
                nan_indices.add(i)
    value_dict = {i: [] for i in nan_indices}

    for x in X:
        for i in nan_indices:
            if not np.isnan(x[i]):
                value_dict[i].append(x[i])

    # compute median values
    median_dict = {i: np.nanmedian(value_dict[i]) if len(value_dict[i]) > 0 else 0 for i in nan_indices}

    # third scan: fill in the nan values with the computed means
    for x in X:
        for i in nan_indices:
            if np.isnan(x[i]):
                x[i] = median_dict[i]
    return X

def normalize_feature_matrix_challenge(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def performance(clf, X, y_true, metric='accuracy'):
    return evaluation.performance(clf, X, y_true, metric=metric)

def cv_performance(clf, X, y, k=5, metric='accuracy'):
    skf = StratifiedKFold(n_splits=k)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)
        score = performance(clf, X_val, y_val, metric=metric)
        scores.append(score)

    return np.mean(scores)

def combined_score(f1_score, auroc, weight):
    return weight * f1_score + (1 - weight) * auroc

def select_C(X, y, C_range, penalties=['l1', 'l2'], k=5, weights=[0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    """ Hyperparameter tuning for C (regularization strength) using cross-validation. """
    skf = StratifiedKFold(n_splits=k)
    best_params = {
        'l1': {'C': None, 'weight': None, 'combined_score': 0},
        'l2': {'C': None, 'weight': None, 'combined_score': 0}
    }
    for penalty in penalties:
        print(f"##### Testing penalty = {penalty} #####")
        for C in C_range:
            for weight in weights:
                f1_scores, auroc_scores = [], []
                # k-fold starts
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    clf = LogisticRegression(penalty=penalty, C=C, solver='liblinear', max_iter=5000, class_weight='balanced')
                    clf.fit(X_train, y_train)
                    
                    f1 = performance(clf, X_val, y_val, metric='f1_score')
                    auroc = performance(clf, X_val, y_val, metric='auroc')
                    
                    f1_scores.append(f1)
                    auroc_scores.append(auroc)
                
                avg_f1 = np.mean(f1_scores)
                avg_auroc = np.mean(auroc_scores)
                
                
                combined = combined_score(avg_f1, avg_auroc, weight)
                
                print(f"C = {C}, weight = {weight}, f1-score = {avg_f1:.4f}, auroc = {avg_auroc:.4f}, combined score = {combined:.4f}")
                
                if combined > best_params[penalty]['combined_score']:
                    best_params[penalty] = {'C': C, 'weight': weight, 'combined_score': combined}
    
    return best_params

    

def run_challenge(X_challenge, y_challenge, X_heldout):
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use generate_challenge_labels to print the predicted labels
    print("================= Part 3 ===================")
    print("Part 3: Challenge")
    # split the data.
    X_train, X_test, y_train, y_test = train_test_split(X_challenge, y_challenge, test_size=0.20, stratify=y_challenge, random_state=3)
    
    C_range = np.logspace(-3, 3, 7)
    best_params = select_C(X_train, y_train, C_range, penalties=['l1', 'l2'], weights=[0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # final evaluation
    print(f"\n########## Best results for l1 ##########")
    C = best_params['l1']['C']
    weight = best_params['l1']['weight']
    print(f"Best C: {C}, Best weight: {weight}, Best combined score: {best_params['l1']['combined_score']:.4f}")
    
    clf = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=5000, class_weight='balanced')
    clf.fit(X_train, y_train)

    f1_train = performance(clf, X_train, y_train, metric='f1_score')
    auroc_train = performance(clf, X_train, y_train, metric='auroc')
    combined_train = combined_score(f1_train, auroc_train, weight)
    print(f"Training F1-score: {f1_train:.4f}, AUROC: {auroc_train:.4f}, Combined score: {combined_train:.4f}")
    
    f1_test = performance(clf, X_test, y_test, metric='f1_score')
    auroc_test = performance(clf, X_test, y_test, metric='auroc')
    combined_test = combined_score(f1_test, auroc_test, weight)
    print(f"Testing F1-score: {f1_test:.4f}, AUROC: {auroc_test:.4f}, Combined score: {combined_test:.4f}")
    
    # final ans
    clf_fin = LogisticRegression(penalty='l1', C=1, solver='liblinear', max_iter=5000, class_weight='balanced')
    clf_fin.fit(X_train, y_train)
    
    y_pred = clf_fin.predict(X_test)
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_challenge, clf_fin.predict(X_challenge)).ravel()
    print("confusion_matrix_test:")
    print(tn, fp, fn, tp)

    
    for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1_score", "auroc", "auprc"]:
        print(metric)
        print(evaluation.performance(clf_fin, X_challenge, y_challenge, metric))
    
    y_score = clf_fin.predict_proba(X_heldout)[:, 1]
    y_label = clf_fin.predict(X_heldout)
    make_challenge_submission(y_label, y_score)



if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()
    run_challenge(X_challenge, y_challenge, X_heldout)
    test_challenge_output()
