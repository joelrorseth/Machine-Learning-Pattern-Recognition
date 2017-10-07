#
# Classifiers
# 60-473 Assignment 1 Q1
#
# Use k-NN classifier with the Euclidean distance function, where k = 1,
# on all four datasets provided.
#

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

import numpy as np
import pandas as pd



def classify_knn(data, k):

    examples_train, examples_test, labels_train, labels_test = data

    # Fit training samples against sample labels
    neighbors = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    neighbors.fit(examples_train, labels_train)

    # Score the training fit compared against the test samples
    print("KNN with k =", k, " classifier score: \t", neighbors.score(examples_test, labels_test))



def classify_naive_bayes(data):

    examples_train, examples_test, labels_train, labels_test = data

    gnb = GaussianNB()
    gnb.fit(examples_train, labels_train)

    print("Gaussian Naive Bayes classifier score:\t", gnb.score(examples_test, labels_test))



def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split into parts
    s = np.split(data, [0, 2, 3], axis=1)

    # Isolate 2D array of feature samples  eg.[[-1.9, 2.4], [...], ...]
    examples = s[1]

    # Change shape of labels from [[1], [1], ... [2], [2]] to [1, 1 .. 2]
    labels = np.reshape(s[2], np.size(s[2]))


    # Try training and testing (NOT 10 fold cross validation)
    # Split array/matrices into random train and test subsets

    return train_test_split(examples, labels, random_state=42)




def main():

    # Uncomment to output before/after each classifier on ALL datasets
    #input_files = ["twogaussians.csv", "halfkernel.csv", \
    #        "twospirals.csv", "clusterincluster.csv"]

    input_files = ["twospirals.csv"]

    for filename in input_files:
        data = split_data(filename)

        print(filename)
        print("-----------------------------------------------")

        print("Data before classification:")
        examples_train, examples_test, labels_train, labels_test = data
        print("Training examples:\n", examples_train)
        print("Training labels:\n", labels_train)
        print("Test examples:\n", examples_test)
        print("Test labels:\n", labels_test)
        print()

        print("After classification:")
        classify_knn(data, 1)
        classify_naive_bayes(data)
        print("\n")


main()
