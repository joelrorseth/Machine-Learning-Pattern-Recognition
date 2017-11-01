#
# Plotting ROC curve
# 60-473 Assignment 2 Q5
#
# For the RBF SVM classifier, plot the ROC curve and determine the
# AUC for each dataset.
#

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SVM Classifier using RBF
def svm_rbf(samples, labels):

    classifier = svm.SVC(kernel='rbf')

    # Use standard train/test split algorithm to obtain distinct sets
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=.5, random_state=0)

    # Use classifier to fit model to training data, then score it with test data
    test_scores = classifier.fit(X_train, y_train).decision_function(X_test)

    # Find True and False Positive rate for ROC curve
    # TODO: I think the positive class should be 1, but it didnt look right
    fpr, tpr, thresholds = roc_curve(y_test, test_scores, pos_label=2)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc



# Plotting function for ROC curve
def plot_roc(name, statistics):

    fpr, tpr, roc_auc = statistics

    plt.figure()

    # Plot ROC curve using FPR and TPR arrays
    # These arrays are of size n, where n is the number of samples
    plt.plot(fpr, tpr, color='green', lw=2,\
            label='ROC curve for Positive class (AUC = %0.2f)' % roc_auc)

    plt.plot([0,1], [0,1], lw=2, linestyle='--')

    # Specify x and y axis range
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Plot labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM RBF Classifier on ' + name)
    plt.legend(loc="lower right")
    plt.show()



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

    input_files = ["twogaussians.csv", "halfkernel.csv", \
            "twospirals.csv", "clusterincluster.csv"]

    # Plot the ROC curve for the SVM RBF classifier on each dataset
    for filename in input_files:

        samples, labels = split_data(filename)
        statistics = svm_rbf(samples, labels)
        plot_roc(filename, statistics)

main()
