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

    km_best_ks = []
    em_best_ks = []

    # Read in and format all 4 datasets into map
    files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    for filename in files:

        # Remove / separate class labels from dataset
        samples, labels = split_data(filename)

        kmeans_best_score = -1
        kmeans_best_k = -1
        em_best_score = -1
        em_best_k = -1

        print("\nEvaluating \'K\' for", filename)

        # Test K-Means clustering model with different k values
        print("----------------------------------")
        print(" K\tK-Means Score\tEM Score")
        print("----------------------------------")

        for k in range(99):

            kmeans_classifier = kmeans(k+2, samples)
            kmeans_pred = kmeans_classifier.fit_predict(samples)

            # Check CH Index for score of clustering model
            km_score = ch_index(samples, kmeans_pred)
            if km_score > kmeans_best_score:
                kmeans_best_score = km_score
                kmeans_best_k = k+2

            em_classifier = em(k+2, samples)
            em_pred = em_classifier.predict(samples)

            # Check CH Index for score of clustering model
            em_score = ch_index(samples, em_pred)
            if em_score > em_best_score:
                em_best_score = em_score
                em_best_k = k+2

            print(" {}\t{:0.4f}\t{:0.4f}".format(k+2, km_score, em_score))


        # Track best results across all k for each dataset
        km_best_ks.append(kmeans_best_k)
        em_best_ks.append(em_best_k)

    print()
    for (i, f) in enumerate(files):
        print("Best k for {}\nKMeans: {}\nEM: {}\n"\
                .format(f, km_best_ks[i], em_best_ks[i]))

main()
