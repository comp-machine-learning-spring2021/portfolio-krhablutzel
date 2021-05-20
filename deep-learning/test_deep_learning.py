import pytest
import deep_learning

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
weather_pd = weather_pd.drop(['DAY', 'STP', 'GUST'], axis=1)
weather_pd['RAIN'] = (weather_pd['PRCP'] > 0).astype(int)

# adapted from original code in project 2
def test_divide_data_shape():
    # calculate sizes
    n = weather_pd.shape[0]
    n_10 = n // 10
    n_90 = n - n_10
    
    # divide data
    test, train = deep_learning.divide_data(weather_pd)
    
    # check sizes
    test_shape = (test.shape[0] == n_10) and (test.shape[1] == weather_pd.shape[1])
    train_shape = (train.shape[0] == n_90) and (train.shape[1] == weather_pd.shape[1])
    
    assert test_shape and train_shape
    
def test_divide_data_type():
    # pandas dataframes
    test, train = deep_learning.divide_data(weather_pd)
    assert isinstance(test, pd.core.frame.DataFrame) and isinstance(train, pd.core.frame.DataFrame)
    
def test_separate_targets_shape():
    # features are (#, 14), targets are (#,)
    test, _ = deep_learning.divide_data(weather_pd)
    feature, target = deep_learning.separate_targets(test)
   
    feature_shape = (feature.shape[0] == test.shape[0] - 1) and (feature.shape[1] == test.shape[1])
    target_shape = (target.shape[0] == test.shape[0] - 1)
    assert feature_shape and target_shape
    
def test_separate_targets_type():
    # returns numpy arrays
    test, _ = deep_learning.divide_data(weather_pd)
    feature, target = deep_learning.separate_targets(test)
    assert isinstance(feature, np.ndarray) and isinstance(target, np.ndarray)
    
def test_build_model_type():
    # builds tf Sequential model
    model = deep_learning.build_model(64, 2)
    assert isinstance(model, tf.keras.Sequential)
    
def test_split_input_target_shape():
    # outputs are 1 shorter than input
    week = np.zeros(5)
    input_days, target_days = deep_learning.split_input_target(week)
    assert len(input_days) == len(target_days) == len(week) - 1
    
def test_split_input_target_type():
    # returns same type outputs as inputs
    week = np.zeros(5)
    input_days, target_days = deep_learning.split_input_target(week)
    assert type(input_days) == type(target_days) == type(week)
        
def test_build_model_RNN_type():
    # builds tf Sequential model
    model = deep_learning.build_model_RNN(2, 256, 16, 4)
    assert isinstance(model, tf.keras.Sequential)

def test_restore_model_type():
    # restores tf Sequential model
    model = deep_learning.restore_model(2, 256, 16)
    assert isinstance(model, tf.keras.Sequential)
    
def test_generate_model_shape():
    # returns 7 days of weather predictions
    expected = 7
    start_weather = np.array([[0, 0, 0, 0, 0, 0, 0]])
    model = deep_learning.restore_model(2, 256, 16)
    days_generated = deep_learning.generate_weather(model, start_weather)
    assert len(days_generated) == expected
    
def test_generate_model_type():
    # returns list of predictions
    start_weather = np.array([[0, 0, 0, 0, 0, 0, 0]])
    model = deep_learning.restore_model(2, 256, 16)
    days_generated = deep_learning.generate_weather(model, start_weather)
    assert isinstance(days_generated, list)
   