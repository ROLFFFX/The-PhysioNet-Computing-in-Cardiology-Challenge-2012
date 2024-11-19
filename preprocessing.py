import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, exceptions

import warnings
warnings.filterwarnings('ignore', category=exceptions.UndefinedMetricWarning)

from helper import *

def generate_feature_vector(df):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Input:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'mean_HR': 84, ...}
    """
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}
    # TODO: Implement this function
    mean_values = {var: [] for var in timeseries_variables} # initialize dict to store timeseries mean
    for index, rows in df.iterrows():
        variable_name = rows['Variable']
        value = rows['Value']
        
        # static variables, use the raw values. 
        # Replace unknown observations (−1) with undefined (use np.nan), 
        # and name these features with the original variable names.
        if variable_name in static_variables:
            if value == -1:
                value = np.nan
            feature_dict[variable_name] = value
            
        # time-series variables, compute the mean of all measurements for that variable. 
        # If no measurement exists for a variable, the mean is also undefined (use np.nan).
        # Name these features as mean {Variable} for each variable.
        elif variable_name in timeseries_variables:
            if value == -1:
                value = np.nan
            mean_values[variable_name].append(value)
    
    for var in timeseries_variables:
        if len(mean_values[var]) > 0:
            feature_dict[f"mean_{var}"] = np.nanmean(mean_values[var])
        else:
            feature_dict[f"mean_{var}"] = np.nan 
    
    # print(feature_dict)
    return feature_dict

def impute_missing_values(X):
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Input:
        X: np.array, shape (N, d). X could contain missing values
    Returns:
        X: np.array, shape (N, d). X does not contain any missing values
    """
    # TODO: Implement this function
    nan_indices = set()
    # first scan: find indices with nan
    for x in X:
        # for each patient's record
        for i in range(0, len(x)):
            val = x[i]
            if (np.isnan(val)):
                nan_indices.add(i)

    # initialize mean_dict to store mean values for nan indices
    sum_dict = {i: 0.0 for i in nan_indices}
    count_dict = {i: 0 for i in nan_indices}


    # second scan: compute sum and count for non-nan values in nan indices
    for x in X:
        for i in nan_indices:
            if not np.isnan(x[i]):
                sum_dict[i] += x[i]
                count_dict[i] += 1

    # compute mean values
    mean_dict = {i: sum_dict[i] / count_dict[i] if count_dict[i] > 0 else 0 for i in nan_indices}

    # third scan: fill in the nan values with the computed means
    for x in X:
        for i in nan_indices:
            if np.isnan(x[i]):
                x[i] = mean_dict[i]
                
    return X


def normalize_feature_matrix(X):
    """
    For each feature column, normalize all values to range [0, 1].

    Input:
        X: np.array, shape (N, d).
    Returns:
        X: np.array, shape (N, d). Values are normalized per column.
    """
    # TODO: Implement this function
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    denominators = max_vals - min_vals
    denominators[denominators == 0] = 1 # replace 0 denominators with 1

    # Normalize the matrix using broadcasting
    X = (X - min_vals) / denominators
    
    # for x in X:
    #     for item in x:
    #         if (np.isnan(item) or item > 1 or item <0):
    #             print("flag")
    #         else:
    #             print(".")

    return X
