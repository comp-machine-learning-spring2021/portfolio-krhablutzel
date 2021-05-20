# Building Seasons for North Carolina
### Unsupervised Learning with k-Means Clustering
import pandas as pd
import numpy as np

# ML tools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering 
from scipy.spatial import distance

# helper functions
# adapted from original code in lab 6
def standardize(data):
    '''standardize each column of a numpy dataframe'''
    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)

    data_std = data.copy()
    for i in range(data.shape[1]): # for every column
        data_std[:,i] = (data[:,i] - mean_vec[i]*np.ones(data.shape[0]))/sd_vec[i]
        
    return data_std

# adapted from original code in Homework 2
def my_kmeans(data, k):
    '''returns labels for k clusters in the data (np array)'''
    # set random start state
    r_state = 101
    
    # start with two random centers
    data_pd = pd.DataFrame(data)
    centers = data_pd.sample(k, random_state = r_state).to_numpy()
    
    verbose = False
    if verbose:
        print(centers)
    
    # one possible stop condition
    maxIter = 300
    
    # each iteration
    for it in range(maxIter):
        if verbose:
            print(it)
        # find distance from center of each cluster
        dists = distance.cdist(data, centers, 'euclidean')

        # assign to cluster
        clusters = np.argmin(dists, axis=1)

        # update centers
        new_centers = np.zeros((k, data.shape[1])) # initialize new centers array
        # for each cluster
        for i in range(len(centers)):
            # take points in cluster
            cluster = data[clusters == i]
            # their avg is new center of cluster
            new_centers[i] = np.mean(cluster, axis = 0)
            
        # print updated centers
        if verbose:
            print(new_centers)

        # stop condition
        if (new_centers == centers).all(): # we didn't change centers
            if verbose:
                print("didn't change centers")
            break
        else:
            centers = new_centers # update centers and continue
            
    # return cluster assignments
    return clusters, centers

# calendar plotting functions
# adapted from https://dzone.com/articles/plotting-a-calendar-in-matplotlib
def plot_calendar(days, months, colors, year):
    '''Build a calendar for {year} given lists for day, month, and color for each day'''
    plt.figure(figsize=(9, 3))
    # colors from https://mycolor.space/
    n_seasons = len(np.unique(colors))
    if n_seasons < 6:
        COLORS = ['#845EC2', '#FF6F91', '#FFC75F', '#9BDE7E', '#039590', '#2F4858',] # more distance btwn colors
    else:
        COLORS = ['#845EC2', '#D65DB1', '#FF6F91', '#FF9671', '#FFC75F', '#F9F871',
                  '#9BDE7E', '#4BBC8E', '#039590', '#1C6E7D', '#2F4858', '#676A8B'] # max 12 colors for now
    ax = plt.gca().axes
    # plot days
    for d, m, c in zip(days, months, colors):
        ax.add_patch(Rectangle((d, m), 
                               width=.8, height=.8, color=COLORS[c]))
    plt.yticks(np.arange(1, 13)+.5, list(calendar.month_abbr)[1:])
    plt.xticks(np.arange(1,32)+.5, np.arange(1,32))
    plt.xlim(1, 32)
    plt.ylim(1, 13)
    plt.title('The {} Seasons of {}'.format(n_seasons, year))
    plt.gca().invert_yaxis()
    # remove borders and ticks
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(top=False, bottom=False, left=False, right=False)
    plt.show()
    
def plot_labels(weather_pd, labels, year):
    '''For a particular year, color the calendar with season labels'''
    # add new season labels to data
    weather_pd['NEW_SEASON'] = labels
    # filter data for year
    weather_year = weather_pd[weather_pd['YEAR'] == year]
    # plot this year's calendar
    plot_calendar(weather_year['DAY'], weather_year['MONTH'], weather_year['NEW_SEASON'], year)
    
# functions for calling sklearn's kmeans/spectral clustering in same format as my_kmeans
# adapted from original code in Homework 3
def kmeans_sklearn(data, k):
    '''returns labels for k clusters in the data'''
    km_alg = KMeans(n_clusters=k)
    fit1 = km_alg.fit(data)
    labels = fit1.labels_
    centers = fit1.cluster_centers_
    return labels, centers

# adapted from original code in Homework 3
def spectral_sklearn(data, k):
    '''returns labels for k clusters in the data'''
    sc_alg = SpectralClustering(n_clusters=k)
    fit1 = sc_alg.fit(data)
    labels = fit1.labels_
    return labels

# adapted from original code in Homework 2
def looping_kmeans(data, kList):
    '''find within cluster sum of squares of k-means for k-values in kList'''
    goodnessList = []
    
    # for each k
    for k in kList:
        # fit k-means for k clusters
        labels, centers = my_kmeans(data, k)
        
        # initialize within cluster sum of squares
        within_cluster_sumsqs = 0

        # loop over clusters
        for c in range(len(centers)):
            # cluster's center and associated points
            cluster_center = [centers[c, :]]
            cluster_points = data[labels == c]

            # distance of each point from the center
            cluster_spread = distance.cdist(cluster_points, cluster_center, 'euclidean')
            
            # total distance of all points from the center (in this cluster)
            cluster_total = np.sum(cluster_spread)

            # add this cluster's within sum of squares to total within_cluster_sumsqs
            within_cluster_sumsqs = within_cluster_sumsqs + cluster_total

        # store goodness for this k
        goodnessList.append(within_cluster_sumsqs)
    
    # goodness for each k
    return goodnessList

def main():
    # import data
    weather_pd = pd.read_csv('../data/weather.csv', index_col = 0)
    # convert to numpy
    weather = weather_pd.drop(['YEAR', 'DAY', 'STP', 'GUST'], axis=1).to_numpy()
    # standardize data
    weather_std = standardize(weather)

    # summarize in 2D
    pca = PCA(n_components=2)
    weather_two = pca.fit_transform(weather_std)

    # run my k-means for 4 seasons
    labels, centers = my_kmeans(weather_std, 4)

    # my implementation of k-means clustering
    labels4, centers = my_kmeans(weather_std, 4)
    labels8, centers = my_kmeans(weather_std, 8)
    labels12, centers = my_kmeans(weather_std, 12)

    # sklearn implementation of k-means clustering
    labels_kmeans_4, centers = kmeans_sklearn(weather_std, 4)
    labels_kmeans_8, centers = kmeans_sklearn(weather_std, 8)
    labels_kmeans_12, centers = kmeans_sklearn(weather_std, 12)

    # sklearn implementation of spectral clustering
    labels_spect = spectral_sklearn(weather_std, 4)

    # test k in [2,20]
    kList = [x for x in range(2, 21)]
    gList = looping_kmeans(weather_std, kList)

    # run k-means for 5 clusters
    labels5, centers = my_kmeans(weather_std, 5)