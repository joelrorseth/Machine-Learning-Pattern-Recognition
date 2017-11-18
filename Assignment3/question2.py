#
# K-Means and EM
# 60-473 Assignment 3 Q2
#
# Pick an index of clustering validity (CH Index), use it to determine the
# best value of K for K-Means and EM on every dataset.
#

from sklearn.metrics import calinski_harabaz_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Create a KMeans clustering object where k=2
def kmeans(k, samples):
    return KMeans(n_clusters=k).fit(samples)


# Create an Expectation Maximization object that uses Gaussian models
def em(k, samples):
    return GaussianMixture(n_components=k).fit(samples)


# Determine the Calinski-Harabaz (CH) Index
def ch_index(samples, clustered_labels):
    return calinski_harabaz_score(samples, clustered_labels)


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

        kmeans_best = -1
        em_best = -1

        print("Evaluating \'K\' for", filename)

        # Test K-Means clustering model with different k values
        print("----------------")
        print("K-Means\nK\tScore")
        print("----------------")

        for k in range(9):

            kmeans_classifier = kmeans(k+2, samples)
            kmeans_pred = kmeans_classifier.fit_predict(samples)

            score = ch_index(samples, kmeans_pred)
            kmeans_best = score if score > kmeans_best

            print("{}\t{:0.4f}".format(k+2, score))

        # Test EM clustering model with different k values
        print("\n----------------")
        print("EM\nK\tScore")
        print("----------------")

        for k in range(9):

            em_classifier = em(k+2, samples)
            em_pred = em_classifier.predict(samples)
            print("{}\t{:0.4f}".format(k+2, ch_index(samples, em_pred)))

        print("\n\n")

main()
