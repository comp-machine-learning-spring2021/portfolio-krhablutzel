# Predicting NC's Existing Seasons
### Supervised Learning with Multiple Classifiers

import pandas as pd
import numpy as np

# ML tools
import itertools
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# helper functions

# adapted from original code in lab 6
def standardize(data):
    '''Standardize a dataframe'''
    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)

    data_std = data.copy()
    for i in range(data.shape[1]): # for every column
        data_std[:,i] = (data[:,i] - mean_vec[i]*np.ones(data.shape[0]))/sd_vec[i]
        
    return data_std

# adapted from original code in project 2
def divide_data(rent):
    '''divide dataset into two sets: 90% test/train and 10% validation'''
    n = rent.shape[0]
    
    # take out 10% of the data for validation
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
    ind_valid = np.random.choice(n, size = n // 10, replace = False)
    rent_valid = rent.iloc[ind_valid]

    # take the other 90% for building the model
    # https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list
    ind_tt = [x for x in range(n) if x not in ind_valid] # not in index
    rent_tt = rent.iloc[ind_tt]

    # shuffle data for test/train so no patterns in folds
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    rent_tt = shuffle(rent_tt)

    return rent_valid, rent_tt

def classification_mse(class_truth, pred_class):
    '''compute classification mse'''
    return np.mean(class_truth != pred_class)

def cross_validation(data, method, k):    
    '''k-fold cross-validation'''
    # calculate fold divisions
    n = data.shape[0]
    n_predictors = data.shape[1] - 1
    foldSize =  n // k # int divide
    foldDivisions = [foldSize * x for x in range(k+1)]

    # adjust for uneven fold size
    if n % k != 0: 
        r = n % k # remainder
        for i in range(1, k+1):
            # add 1 + previous size increase to each group until r
            # then just shift by r to account for previous size increases
            foldDivisions[i] += min(i, r)

    # divide into folds
    folds = []
    for i in range(k):
        folds.append(data.iloc[foldDivisions[i]:foldDivisions[i+1], :])

    # linear model w/ each fold as test once
    test_errors = []

    for i in range(k):
        # get test fold
        test = folds[i]

        # combine other folds into training set
        train_folds = folds.copy()
        train_folds.pop(i)
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        train = pd.concat(train_folds) # concatenate folds

        if method == 'SVM':
            # fit SVM to training data
            mod = SVC(kernel="rbf")
        elif method == 'neighbor':
            # fit kNN to training data
            mod = KNeighborsClassifier(n_neighbors = 9)
        elif method == 'tree':
            # fit decision tree to training data
            mod = DecisionTreeClassifier()
        elif method == 'forest':
            # build random forest classifier for training data
            mod = RandomForestClassifier(n_estimators=10, max_features = min(3, n_predictors), max_depth=3, random_state=0)
        
        mod.fit(train.iloc[:,:-1], train.iloc[:,-1]) # class var in last column
        
        # compute testing error
        test_preds = mod.predict(test.iloc[:,:-1])
        test_error = classification_mse(test_preds, test.iloc[:,-1])
        test_errors.append(test_error)

    # cross validation error - avg of test errors
    cross_val_error = np.mean(test_errors)
    
    return cross_val_error

def all_cv_errors(rent_tt, methods):
    '''get cross-validation error for all possible models'''
    cv_errors = []     
    # test each possible model type
    for method in methods:
        # compute cross-validation error
        cv_err = cross_validation(rent_tt, method, 10)

        # store errors
        cv_errors.append([method, cv_err])
                
    # sort cv errors w/ lowest in first row
    cv_err_np = np.array(cv_errors, dtype=object)
    cv_err_np = cv_err_np[np.argsort(cv_err_np[:,1])]
    return cv_err_np

def main():
    # import data
    weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
    weather_pd = weather_pd.drop(['YEAR', 'DAY', 'STP', 'GUST'], axis=1) # remove faulty columns and some temporal indicators

    # move seasons (classes) to last column
    columns = list(weather_pd.columns)
    columns.remove('SEASON')
    columns.append('SEASON')
    weather_pd = weather_pd[columns]

    # summarize in 2D
    pca = PCA(n_components=2)
    weather_two = pca.fit_transform(standardize(weather_pd.to_numpy()))

    # separate data into test/train and validation sets
    np.random.seed(888)
    weather_valid, weather_tt = divide_data(weather_pd)

    # get all cv errors for each method
    cv_errors = all_cv_errors(weather_tt, ['neighbor', 'tree', 'forest']) # excluding 'SVM'

    # store best method
    best_method = cv_errors[0,0]
    lowest_cv_err = cv_errors[0,1]

    # build best type of model
    dt = DecisionTreeClassifier()
    dt.fit(weather_tt.iloc[:,:-1], weather_tt.iloc[:,-1]) # from all but the last column, predict last column

    # compute validation error
    val_preds = dt.predict(weather_valid.iloc[:,:-1])
    val_error = classification_mse(val_preds, weather_valid.iloc[:,-1])
