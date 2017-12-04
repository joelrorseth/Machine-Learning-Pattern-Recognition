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


# Data formatting
def split_data(filename):

    data = pd.read_csv(filename, header=0).as_matrix()

    # Split input CSV into parts
    # s[0]-2D array of labels , s[1]-2D array of sample data

    s = np.split(data, [1], axis=1)

    return s[1], np.reshape(s[0], np.size(s[0]))


def main():

    filename = "Bladder cancer gene expressions.csv"
    samples, labels = split_data(filename)

    #tn, fp, fn, tp = classifier(samples, labels)
    #print("The accuracy of ___  is", accuracy(tn, fp, fn, tp))

    print()

main()
