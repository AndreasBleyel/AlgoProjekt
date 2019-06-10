#!/usr/bin/env python
# coding: utf-8

# ## Helper Funktionen

# In[ ]:


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

def checkFeature(feature):
    checkNAs(feature)
    checkForNegatives(feature)
    overview(feature)
    plotDistribution(feature)
    if feature != target:
        plotRelationToTarget(feature)
    
def checkNAs(feature):
    if train[feature].isna().sum() > 0:
        print(bcolors.FAIL + "Sum NAs: " + str(train[feature].isna().sum()))
    else:
        print(bcolors.OKGREEN + "No NAs" +bcolors.ENDC)

def checkForNegatives(feature):
    if any(train[feature]<0):
        print (bcolors.WARNING + "Warning feature has negative value!" + bcolors.ENDC)
    else:
        print (bcolors.OKGREEN + "No negative values" + bcolors.ENDC)

def plotDistribution(feature):
    sns.distplot(train[feature], fit=norm);
    fig = plt.figure()
    res = stats.probplot(train[feature], plot=plt)
    
def plotRelationToTarget(feature):
    data_temp = pd.concat([train[target], train[feature]], axis=1)
    data_temp.plot.scatter(x=feature, y=target, ylim=(0,800000));
        
def overview(feature):
    print(train[feature].describe())
    print(bcolors.HEADER + "Head" +bcolors.ENDC)
    print(train[feature].head(3))
    
def printSkewKurt(feature):
    print("Skewness: %f" % train[feature].skew())
    print("Kurtosis: %f" % train[feature].kurt())
    
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

