#!/usr/bin/env python
# coding: utf-8

# Helper Funktionen

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import math
from sklearn.metrics import mean_squared_error


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
target = 'SalePrice'

def checkFeature(feature, data, is_cat):
    checkNAs(feature, data)
    checkForNegatives(feature, data)
    overview(feature, data)
    plotDistribution(feature, data, is_cat)
    if feature != target:
        plotRelationToTarget(feature, data)
    
def checkNAs(feature, data):
    if data[feature].isna().sum() > 0:
        print(bcolors.FAIL + "Sum NAs: " + str(data[feature].isna().sum()))
    else:
        print(bcolors.OKGREEN + "No NAs" +bcolors.ENDC)

def checkForNegatives(feature, data):
    if any(data[feature]<0):
        print (bcolors.WARNING + "Warning feature has negative value!" + bcolors.ENDC)
    else:
        print (bcolors.OKGREEN + "No negative values" + bcolors.ENDC)

def plotDistribution(feature, data, is_cat):
    sns.distplot(data[feature], fit=norm);
    fig = plt.figure()
    if not is_cat:
        res = stats.probplot(data[feature], plot=plt)
    
def plotRelationToTarget(feature, data):
    data_temp = pd.concat([data[target], data[feature]], axis=1)
    data_temp.plot.scatter(x=feature, y=target, ylim=(0,800000));
        
def overview(feature, data):
    print(data[feature].describe())
    print(bcolors.HEADER + "Head" +bcolors.ENDC)
    print(data[feature].head(3))
    
def printSkewKurt(feature, data):
    print("Skewness: %f" % data[feature].skew())
    print("Kurtosis: %f" % data[feature].kurt())
    
def calculate_performance(prediction, actual, scaler):
    if scaler == True:
        p = scaler.inverse_transform(prediction.reshape(-1,1))
        a = scaler.inverse_transform(actual.reshape(-1,1))
    else:
        p = prediction
        a = actual
        
    mse = mean_squared_error(a, p)
    err = np.sqrt(mse)
    r2 = r2_score(a, p)
    mae = median_absolute_error(a, p)
    
    return (mse, err, r2, mae)

def print_performance(measure_tuple):
    
    mse = measure_tuple[0]
    err = measure_tuple[1]
    r2 = measure_tuple[2]
    mae = measure_tuple[3]
    
    print("Mean squared error is {}".format(str(mse)))
    print("Positive mean error is {}".format(str(err)))
    print("Overall R² is {}".format(str(r2)))
    print("Median absolute error is {}".format(str(mae)))

def eval_model(model, test_X, test_y):
    r2 = model.score(test_X, test_y)

    pred_y = model.predict(test_X)
    rmse = math.sqrt(mean_squared_error(np.exp(test_y), np.exp(pred_y)))
    
    print('r2 = ' + str(r2))
    print('rmse = ' + str(rmse))
    return rmse, r2