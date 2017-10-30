#
# Testing the SVM Classifiers
# 60-473 Assignment 2 Q3
#
# Run all three variations of the SVM classifier on all four datasets,
# using 10 fold cross validation. For each of the three classifiers,
# output their average PPV, NPV, specificity, sensitivity and accuracy
# performance across the datasets.
#

from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd


# SVM Classifier using linear kernel
def svm_linear(samples, labels):

    classifier = svm.SVC(kernel='linear')
    #classifier.fit(samples, labels)

    # Use confusion matrix to extract statistics on cross validated prediction
    prediction = cross_val_predict(classifier, samples, labels, cv=10)
    statistics = confusion_matrix(labels, prediction).ravel()

    return calculate_efficiency(statistics)



# SVM Classifier using deg. 2 polynomial kernel
def svm_polynomial(samples, labels):

    classifier = svm.SVC(kernel='poly', degree=2)
    #classifier.fit(samples, labels)

    # Use confusion matrix to extract statistics on cross validated prediction
    prediction = cross_val_predict(classifier, samples, labels, cv=10)
    statistics = confusion_matrix(labels, prediction).ravel()

    return calculate_efficiency(statistics)



# SVM Classifier using RBF
def svm_rbf(samples, labels):

    classifier = svm.SVC(kernel='rbf')
    #classifier.fit(samples, labels)

    # Use confusion matrix to extract statistics on cross validated prediction
    prediction = cross_val_predict(classifier, samples, labels, cv=10)
    statistics = confusion_matrix(labels, prediction).ravel()

    return calculate_efficiency(statistics)


# Return 5-tuple corresponding to the 5 calculated measures of efficiency
def calculate_efficiency(statistics):

    tn, fp, fn, tp = statistics

    # Score the training fit compared against the test samples
    return accuracy(tn, fp, fn, tp), sensitivity(tp, fn),\
            specificity(tn, fp), ppv(tp, fp), npv(tn, fn)


# Print all measures out nicely
def print_efficiency(acc, sens, spec, ppv, npv):

    print("Accuracy:   \t", acc)
    print("Sensitivity:\t", sens)
    print("Specificity:\t", spec)
    print("PPV:        \t", ppv)
    print("NPV:        \t", npv)
    print()


# Proportion of successful predictions
def accuracy(tn, fp, fn, tp):
    return ((tp+tn) / float(tp+fp+tn+fn))

# Of all that should be classified as Positive, how many were?
def sensitivity(tp, fn):
    return (tp / float(tp+fn))

# Of all that should be classified as Negative, how many were?
def specificity(tn, fp):
    return (tn / float(tn+fp))

# Of all classified as Positive, how many were correctly classified?
def ppv(tp, fp):
    return (tp / float(tp+fp))

# Of all classified as Negative, how many were correcly classified?
def npv(tn, fn):
    return (tn / float(tn+fn))




# Read CSV and separate into samples and labels
def split_data(filename):

    # Read data into 2D array of samples  eg. [[-1.9, 2.4, 1], [...], ...]
    data = pd.read_csv(filename, header=None).as_matrix()

    # Split input CSV into parts
    # s[0]-Empty , s[1]-2D array of sample data , s[2]-2D array of labels

    s = np.split(data, [0, 2, 3], axis=1)
    return s[1], np.reshape(s[2], np.size(s[2]))



def main():

    classifier_names = ["SVM Classifier (Linear Kernel)",\
            "SVM Classifier (Polynomial Kernel, Degree = 2)",
            "SVM Classifier (RBF)"]
    input_files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    counter = 0

    # Read in and format all 4 datasets into map
    filedata = {}
    for filename in input_files:
        filedata[filename] = split_data(filename)

    # Run each SVM classifier
    for classifier in [svm_linear, svm_polynomial, svm_rbf]:
        acc = 0.0
        sen = 0.0
        spec = 0.0
        ppv = 0.0
        npv = 0.0

        print("Now running", classifier_names[counter])
        print()
        counter += 1

        # Test current classifier against every CSV
        for filename in input_files:

            # Grab data for this CSV
            samples, labels = filedata[filename]

            print("Evaluating", filename + "...")

            # Call the classifier, add results to others from other files
            t_acc, t_sen, t_spec, t_ppv, t_npv = classifier(samples, labels)
            acc += t_acc
            sen += t_sen
            spec += t_spec
            ppv += t_ppv
            npv += t_npv

        # Compute and print averages for each measure of efficiency
        acc /= 4
        sen /= 4
        spec /= 4
        ppv /= 4
        npv /= 4

        print("\nSummary\n---------------------------------")
        print_efficiency(acc, sen, spec, ppv, npv)
        print()

main()
