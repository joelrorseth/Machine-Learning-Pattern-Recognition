#
# Testing the SVM Classifiers
# 60-473 Assignment 2 Q2
#
# Test out the linear, polynomial and RBF kernel SVM classifiers.
#

from sklearn import svm

import numpy as np
import pandas as pd


# SVM Classifier using linear kernel
def svm_linear(samples, labels):

    classifier = svm.SVC(kernel='linear')
    classifier.fit(samples, labels)


# SVM Classifier using deg. 2 polynomial kernel
def svm_polynomial(samples, labels):

    classifier = svm.SVC(kernel='poly', degree=2)
    classifier.fit(samples, labels)


# SVM Classifier using RBF
def svm_rbf(samples, labels):

    classifier = svm.SVC(kernel='rbf')
    classifier.fit(samples, labels)



# Read CSV and separate into samples and labels
def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split input CSV into parts
    # s[0]-Empty , s[1]-2D array of sample data , s[2]-2D array of labels

    s = np.split(data, [0, 2, 3], axis=1)
    return s[1], np.reshape(s[2], np.size(s[2]))



def main():

    input_files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    for filename in input_files:

        samples, labels = split_data(filename)

        # Run with all 3 classifiers
        svm_linear(samples, labels)
        svm_polynomial(samples, labels)
        svm_rbf(samples, labels)

main()
