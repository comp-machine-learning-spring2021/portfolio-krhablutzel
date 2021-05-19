# tests adapted from Homework 2
import pytest
import pandas as pd
import numpy as np
import clustering

# import data
weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
weather = weather_pd.drop(['YEAR', 'DAY', 'STP', 'GUST'], axis=1).to_numpy()
weather_std = clustering.standardize(weather)

def test_standardize_type():
    # returns np dataframe
    assert isinstance(clustering.standardize(weather), np.ndarray)
    
def test_standardize_size():
    # returns dataframe of the same size
    expected = weather.shape
    assert expected == clustering.standardize(weather).shape
    
def test_standardize_mean_centered():
    # column mean is near 0
    expected = weather.shape[1]
    near0 = np.isclose(np.mean(clustering.standardize(weather), axis=0), 0)
    assert expected == sum(near0)

def test_my_kmeans_type():
    # returns tuple
    assert isinstance(clustering.my_kmeans(weather_std, 4), tuple)

def test_my_kmeans_shape():
    # returns tuple of length 2
    expected = 2
    assert len(clustering.my_kmeans(weather_std, 4)) == expected

def test_my_kmeans_center_num():
    # centers shaped as expected
    expected = (4,12)
    centers_shape = clustering.my_kmeans(weather_std, 4)[1].shape
    assert centers_shape == expected

def test_my_kmeans_labels():
    # labels 0-3 for k=4
    expected = 3
    label_max = np.max(clustering.my_kmeans(weather_std, 4)[0])
    assert label_max == expected
    
def test_my_kmeans_labels_size():
    # one label per datapoint
    expected = weather_std.shape[0]
    label_len = len(clustering.my_kmeans(weather_std, 4)[0])
    assert label_len == expected

def test_my_kmeans_different_cols():
    # cols are different in centers
    # tests first 2 cols
    expected = False
    centers=clustering.my_kmeans(weather_std, 4)[1]
    comp_cols = sum(centers[:,0] == centers[:,1]) == 4
    assert comp_cols == expected

def test_kmeans_sklearn_type():
    # returns tuple
    assert isinstance(clustering.kmeans_sklearn(weather_std, 4), tuple)

def test_kmeans_sklearn_shape():
    # returns tuple of length 2
    expected = 2
    assert len(clustering.kmeans_sklearn(weather_std, 4)) == expected

def test_kmeans_sklearn_center_num():
    # centers shaped as expected
    expected = (4,12)
    centers_shape = clustering.kmeans_sklearn(weather_std, 4)[1].shape
    assert centers_shape == expected

def test_kmeans_sklearn_labels():
    # labels 0-3 for k=4
    expected = 3
    label_max = np.max(clustering.kmeans_sklearn(weather_std, 4)[0])
    assert label_max == expected
    
def test_kmeans_sklearn_labels_size():
    # one label per datapoint
    expected = weather_std.shape[0]
    label_len = len(clustering.kmeans_sklearn(weather_std, 4)[0])
    assert label_len == expected

def test_kmeans_sklearn_different_cols():
    # cols are different in centers
    # tests first 2 cols
    expected = False
    centers=clustering.kmeans_sklearn(weather_std, 4)[1]
    comp_cols = sum(centers[:,0] == centers[:,1]) == 4
    assert comp_cols == expected

def test_spectral_shape():
    # returns list of labels as long as the data
    expected = weather_std.shape[0]
    assert len(clustering.spectral_sklearn(weather_std, 4)) == expected

def test_spectral_labels():
    # labels 0-3 for k=4 (or fewer if doesn't find all 4 clusters)
    expected = 3
    label_max = np.max(clustering.spectral_sklearn(weather_std, 4))
    assert label_max <= expected

def test_looping_kmeans_type():
    # returns list
    assert isinstance(clustering.looping_kmeans(weather_std,
        [x for x in range(2, 21)]), list)

def test_looping_kmeans_size():
    # returns list of correct size
    expected = 19
    assert len(clustering.looping_kmeans(weather_std, [x for x in range(2, 21)])) == expected

def test_looping_kmeans_goodness():
    # distance decreases as k increases
    out = clustering.looping_kmeans(weather_std,[x for x in range(2, 21)])
    assert (out[1:] <= out[:-1])