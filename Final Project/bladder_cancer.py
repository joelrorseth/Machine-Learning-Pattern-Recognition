#
# Bladder Cancer Subtypes Classification
# 60-473 Final Project
#
# Authors:
# - Joel Rorseth
# - Michael Bianchi
#
# The problem consists of finding meaningful biomarkers for different bladder
# cancer stages. This can be done via classification and feature selection for
# selecting genes that contribute to one or more different classifications
# between stages or among all stages. The dataset contains several samples that
# belong to different stages. The stage (class) is given in the first column.
# All other columns contain expressions (features) for many different genes. You
# are free to work on one or more stages as follows: classification, solving the
# multi-class problem, feature selection, other aspects, or a combination of
# these. For the multiclass problem, you can consider 3 classes: T1, T2 and Ta.
#

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

import numpy as np
import pandas as pd


'''
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
'''

def calculate_efficiency(classifier, samples, labels):

    pred = cross_val_predict(classifier, samples, labels, cv=5)
    score = cross_val_score(classifier, samples, labels, cv=5, scoring="accuracy")

    #print("Accuracy:", score)
    print("Average accuracy over CV runs:", np.average(score))

    conf_matrix = confusion_matrix(labels, pred)
    print("Confusion Matrix:\n", conf_matrix)



# SVM Classifier
def svm_classifier():
    return svm.SVC(kernel='rbf', C=1, gamma=1)

# Random Forest
def rf_classifier():
    return RandomForestClassifier()


# Data formatting
def split_data(filename):

    data = pd.read_csv(filename, header=0).as_matrix()

    # TODO: There appears to be 4 superclasses: Ta, T1, T2 and Ti
    # TODO: For now we will throw out those samples classified as Ti
    # TODO: There is too few Ti samples (2), doesnt work out nicely

    def legit(a):
        return not a[0].startswith('Ti')

    bool_arr = np.array([ legit(row) for row in data ])
    filtered = data[bool_arr]


    # Split input CSV into parts
    # s[0]-2D array of labels , s[1]-2D array of sample data

    s = np.split(filtered, [1], axis=1)

    #samples = np.array(filter(lambda a: !a[0].startswith('Ti'), s[1]))

    # Simplify all labels to their inherent groupings (eg. T1, T2, Ta)
    labels = np.reshape(s[0], np.size(s[0]))
    labels = [cancer[:2] for cancer in labels]

    # Uncomment to see sample size for each class
    #c = Counter(labels)
    #print(c)

    return s[1], labels


def main():

    filename = "Bladder cancer gene expressions.csv"
    samples, labels = split_data(filename)

    print("Testing with SVM Classifier")
    c = svm_classifier()
    calculate_efficiency(c, samples, labels)

    print()
    print("Testing with Random Forest Classifier")
    r = rf_classifier()
    calculate_efficiency(r, samples, labels)

main()
