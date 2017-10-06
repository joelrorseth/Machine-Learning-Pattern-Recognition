#
# Classifiers (10 Fold Cross Validation)
# 60-473 Assignment 1 Q2
#
# Use k-NN classifier with the Euclidean distance function, where k = 1,
# on all four datasets provided. 10 Fold Cross Validation must be used to
# evaulate the classifications on each dataset, and efficiency measures
# must be reported.
#

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd



# Proportion of successful predictions
def accuracy(tn, fp, fn, tp):
    return ((tp+tn) / float(tp+fp+tn+fn))

# Proportion of correctly identified positives (tp)
def sensitivity(tp, fn):
    return (tp / float(tp+fn))

# Proportion of correctly identified negatives (tn)
def specificity(tn, fp):
    return (tn / float(tn+fp))

# (Precision) - Proportion of correctly identified positives
def ppv(tp, fp):
    return (tp / float(tp+fp))

# (Recall) - Proportion of correctly identified negatives
def npv(tn, fn):
    return (tn / float(tn+fn))




def classify_knn(examples, labels, k):

    # Fit training samples against sample labels
    neighbors = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Use confusion matrix to extract statistics on cross validated prediction
    prediction = cross_val_predict(neighbors, examples, labels, cv=10)
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()

    # Score the training fit compared against the test samples
    print("K Nearest Neighbor Classifier with k =", k)
    print("Accuracy:   \t", accuracy(tn, fp, fn, tp))
    print("Sensitivity:\t", sensitivity(tp, fn))
    print("Specificity:\t", specificity(tn, fp))
    print("PPV:        \t", ppv(tp, fp))
    print("NPV:        \t", npv(tn, fn))



# TODO
def classify_naive_bayes(examples, labels):

    print()
    #gnb = GaussianNB()



def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split into parts
    s = np.split(data, [0, 2, 3], axis=1)

    # Isolate 2D array of feature samples  eg.[[-1.9, 2.4], [...], ...]
    samples = s[1]

    # Isolate 2D array of labels  eg. [[1], [1], ... [2], [2]]
    labels = s[2]

    # Consolidate labels 2D array into one big 1D array
    reformatted_labels = np.reshape(labels, np.size(labels))

    return samples, reformatted_labels




def main():

    input_files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    for filename in input_files:
        examples, labels = split_data(filename)

        print(filename)

        classify_knn(examples, labels, 1)
        classify_naive_bayes(examples, labels)
        print()


main()
