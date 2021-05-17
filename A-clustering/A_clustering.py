import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering 
from scipy.spatial import distance

# code from hw3
def kmeans_sklearn(data, k):
    km_alg = KMeans(n_clusters=k)
    fit1 = km_alg.fit(data)
    labels = fit1.labels_
    #centers = fit1.cluster_centers_
    
    return labels #, centers

def spectral_sklearn(data, k):
    sc_alg = SpectralClustering(n_clusters=k)
    fit1 = sc_alg.fit(data)
    labels = fit1.labels_
    
    return labels

# Question 2
def make_adj(array_np):
    pair_dists = distance.cdist(array_np, array_np, 'euclidean')
    
    # adapted from:
    # https://www.geeksforgeeks.org/python-replace-negative-value-with-zero-in-numpy-array/
    pair_dists[pair_dists < 0.5] = 0.3 # placeholder so doesn't get set to 0
    pair_dists[pair_dists >= 0.5] = 0 
    pair_dists[pair_dists > 0] = 1 # undo placeholder
    
    # make sure diagonals are 0
    # from https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
    np.fill_diagonal(pair_dists, 0)
    
    return pair_dists
    
def my_laplacian(A):
    '''returns unnormalized Laplacian of numpy array/adjacency list'''
    # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    rowSums = np.sum(A, axis = 1)

    # degree matrix
    # counts num datapoints each datapoint near
    # https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    D = np.diag(rowSums)

    # unnormalized Laplacian
    L = D - A

    return L

# Question 3
def spect_clustering(L, k):
    # eigens of L
    eig_vals, eig_vecs = np.linalg.eig(L)
    # prevent complex numbers
    # https://numpy.org/doc/stable/reference/generated/numpy.real.html
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    print("eigvals", eig_vals)
    print("eigvecs", eig_vecs)
    
    # sort smallest to greatest eigenvalue
    inds = eig_vals.argsort()
    eig_vals = eig_vals[inds]
    eig_vecs = eig_vecs[:,inds]
    print("eigvec shape", eig_vecs.shape)
    
    # find non-zero eigenvalues only
    # https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    # pointed me to np.nonzero()
    nonZinds = np.nonzero(eig_vals)
    eig_vals = eig_vals[nonZinds]
    eig_vecs = eig_vecs[:,nonZinds]
    
    # first k eigenvectors
    k_eig_vecs = eig_vecs[:, :k]
    
    print("keigvec shape", k_eig_vecs.shape)
    
    return kmeans_sklearn(k_eig_vecs, k)

# code from lab 6
def standardize(data):
    '''Standardize a dataframe'''
    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)

    data_std = data.copy()
    for i in range(data.shape[1]): # for every column
        data_std[:,i] = (data[:,i] - mean_vec[i]*np.ones(data.shape[0]))/sd_vec[i]
        
    return data_std

def import_data():
    # import data
    weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
    weather_pd = weather_pd.drop(['YEAR', 'MONTH', 'DAY', 'SEASON', 'STP', 'GUST'], axis=1) # remove temporal indicators

    # convert to numpy
    weather = weather_pd.to_numpy()
    weather_std = standardize(weather)
    return weather_std

def main():
    weather_std = import_data()

    # Find Clusters with Spectral Clustering
    A = make_adj(weather_std)
    L = my_laplacian(A)
    labels_my_spect = spect_clustering(L, 2)

    ## sklearn implementations
    # K-Means standardized
    labels_kmeans = kmeans_sklearn(weather_std, 4)

    # Spectral Clustering standardized
    labels_spect = spectral_sklearn(weather_std, 4)
    
main()
