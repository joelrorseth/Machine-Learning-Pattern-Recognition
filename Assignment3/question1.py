#
# K-Means and EM
# 60-473 Assignment 3 Q1
#
# Run K-Means (with Euclidean distance) and Expectation Maximization
# (EM), where k = 2 for both algorithms.
#

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd


# Create a KMeans clustering object where k=2
def kmeans(samples):
    return KMeans(n_clusters=2).fit(samples)

# Create an Expectation Maximization object that uses Gaussian models
def em(samples):
    return GaussianMixture(n_components=2).fit(samples)



# Read CSV and separate into samples and labels
def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split input CSV into parts
    # s[0]-Empty , s[1]-2D array of sample data , s[2]-2D array of labels

    s = np.split(data, [0, 2, 3], axis=1)
    return s[1], np.reshape(s[2], np.size(s[2]))

def main():

    # Read in and format all 4 datasets into map
    for filename in ["twogaussians.csv", "halfkernel.csv",
            "twospirals.csv", "clusterincluster.csv"]:

        # Remove / separate class labels from dataset
        samples, labels = split_data(filename)

        # Instantiate classifier, calculate and print efficiency
        kmeans_classifier = kmeans(samples)
        em_classifier = em(samples)

    print()

main()
