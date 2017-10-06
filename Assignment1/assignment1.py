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

    coord_train, coord_test, class_train, class_test = data

    # Fit training samples against sample labels
    neighbors = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    neighbors.fit(coord_train, class_train)

    # Score the training fit compared against the test samples
    print("KNN with k =", k, " classifier score: ", neighbors.score(coord_test, class_test))



def classify_naive_bayes(data):

    coord_train, coord_test, class_train, class_test = data

    gnb = GaussianNB()
    gnb.fit(coord_train, class_train)

    print("Gaussian Naive Bayes classifier score: ", gnb.score(coord_test, class_test))



def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split into parts
    s = np.split(data, [0, 2, 3], axis=1)

    # Isolate 2D array of feature samples  eg.[[-1.9, 2.4], [...], ...]
    coordinates = s[1]

    # Isolate 2D array of labels  eg. [[1], [1], ... [2], [2]]
    labels = s[2]

    # Consolidate labels 2D array into one big 1D array
    classes = np.reshape(labels, np.size(labels))


    # Try training and testing (NOT 10 fold cross validation)
    # Split array/matrices into random train and test subsets

    return train_test_split(coordinates, classes, random_state=42)




def main():

    input_files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    for filename in input_files:
        data = split_data(filename)

        print(filename)

        for i in range(7):
            classify_knn(data, i+1)
        classify_naive_bayes(data)
        print()


main()
