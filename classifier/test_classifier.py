import pytest
import pandas as pd
import numpy as np
import classifier

# ML tools
import itertools
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# import data
weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
weather_pd = weather_pd.drop(['YEAR', 'DAY', 'STP', 'GUST'], axis=1) # remove faulty columns and some temporal indicators

# move seasons (classes) to last column
columns = list(weather_pd.columns)
columns.remove('SEASON')
columns.append('SEASON')
weather_pd = weather_pd[columns]

def test_standardize_type():
    # returns np dataframe
    assert isinstance(classifier.standardize(weather_pd.to_numpy()), np.ndarray)
    
def test_standardize_size():
    # returns dataframe of the same size
    expected = weather_pd.shape
    assert expected == classifier.standardize(weather_pd.to_numpy()).shape
    
def test_standardize_mean_centered():
    # column mean is near 0
    expected = weather_pd.shape[1]
    near0 = np.isclose(np.mean(classifier.standardize(weather_pd.to_numpy()), axis=0), 0)
    assert expected == sum(near0)

# adapted from original code in project 2
def test_divide_data_shape():
    # calculate sizes
    n = weather_pd.shape[0]
    n_10 = n // 10
    n_90 = n - n_10
    
    # divide data
    valid, tt = classifier.divide_data(weather_pd)
    
    # check sizes
    val_shape = (valid.shape[0] == n_10) and (valid.shape[1] == weather_pd.shape[1])
    tt_shape = (tt.shape[0] == n_90) and (tt.shape[1] == weather_pd.shape[1])
    
    assert val_shape and tt_shape
    
def test_divide_data_type():
    # pandas dataframes
    valid, tt = classifier.divide_data(weather_pd)
    assert isinstance(valid, pd.core.frame.DataFrame) and isinstance(tt, pd.core.frame.DataFrame)
    
def test_classification_mse_type():
    # expected type
    class_truth = pd.DataFrame([1, 1, 0, 0])
    pred_class = pd.DataFrame([0, 1, 0, 1])
    assert isinstance(classifier.classification_mse(class_truth, pred_class)[0], float)
    
def test_classification_mse_vals():
    # correct mse
    expected = 0.5
    class_truth = pd.DataFrame([1, 1, 0, 0])
    pred_class = pd.DataFrame([0, 1, 0, 1])
    assert classifier.classification_mse(class_truth, pred_class)[0] == expected
    
def test_cross_validation_type():
    # returns float
    out = classifier.cross_validation(weather_pd, 'tree', 10)
    assert isinstance(out, float)
    
def test_cross_validation_methods():
    # runs successfully for all three methods
    tree = classifier.cross_validation(weather_pd, 'tree', 10)
    kNN = classifier.cross_validation(weather_pd, 'neighbor', 10)
    forest = classifier.cross_validation(weather_pd, 'forest', 10)
    assert isinstance(tree, float) and isinstance(kNN, float) and isinstance(forest, float)
    
def test_all_cv_errors_shape():
    # returns expected number of combinations and cols
    out = classifier.all_cv_errors(weather_pd, ['neighbor', 'tree', 'forest'])
    expected = (3, 2)
    assert out.shape == expected
    
def test_all_cv_errors_type():
    # returns expected types in each column
    out = classifier.all_cv_errors(weather_pd, ['neighbor', 'tree', 'forest'])
    assert isinstance(out[0,0], str) and isinstance(out[0,1], float)
    
def test_all_cv_errors_col3_sorted():
    # second column is sorted ascending
    out = classifier.all_cv_errors(weather_pd, ['neighbor', 'tree', 'forest'])
    assert out[0,1] <= out[1,1] and out[1,1] <= out[2,1]
