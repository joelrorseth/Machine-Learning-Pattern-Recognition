#
# Classifiers
# 60-473 Assignment 1 Q4
#
# Plot the datasets
#

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



def classify_knn(title, i, data, k):

    examples_train, examples_test, labels_train, labels_test = data

    # Fit training samples against sample labels
    neighbors = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    neighbors.fit(examples_train, labels_train)

    styles = ['>', '<', 's', 'o']
    colors = ['b', 'r', 'g', 'y']

    # Plot data from dataset
    X = examples_train
    y = labels_train

    for index in range(0, len(X)):
        if y[index] == 1:
            plt.scatter(X[index, 0], X[index, 1], c=colors[i%4], s=8, marker=styles[i%4])
        else:
            plt.scatter(X[index, 0], X[index, 1], c=colors[(i+1)%4], s=8, marker=styles[(i+1)%4])

    plt.title(title)
    plt.show()




def classify_naive_bayes(data):

    examples_train, examples_test, labels_train, labels_test = data

    gnb = GaussianNB()
    gnb.fit(examples_train, labels_train)
    #print("Gaussian Naive Bayes classifier score:\t", gnb.score(examples_test, labels_test))



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
    input_files = ["twogaussians.csv", "halfkernel.csv", "twospirals.csv", "clusterincluster.csv"]
    counter = 0

    for filename in input_files:
        data = split_data(filename)

        print(filename)
        print("-----------------------------------------------")

        classify_knn(filename, counter, data, 1)
        classify_naive_bayes(data)
        print("\n")
        counter += 1


main()
