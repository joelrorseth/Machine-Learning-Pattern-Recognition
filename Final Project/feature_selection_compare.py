#
# Bladder Cancer Subtypes Classification
# 60-473 Final Project
#
# Authors:
# - Joel Rorseth
# - Michael Bianchi
#
# In this script, we establish a test to run several classifiers on the original
# dataset, then compare the results of running it again on various feature
# selected datasets.
#

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm, tree, ensemble, preprocessing

from sklearn.feature_selection import SelectPercentile, mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression
from sklearn.feature_selection import mutual_info_regression
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


# Run tests and output stats for given classification
def calculate_efficiency(classifier, samples, labels):

    pred = cross_val_predict(classifier, samples, labels, cv=5)
    score = cross_val_score(classifier, samples, labels, cv=5, scoring="accuracy")

    print("Average accuracy over CV runs:", np.average(score))

    # Print confusion matrix and other stats
    #print(classification_report(labels, pred))

    conf_matrix = confusion_matrix(labels, pred)
    print("Confusion Matrix:\n", conf_matrix)



# SVM Classifier
def svm_classifier():
    return svm.SVC(kernel='rbf', C=1, gamma=1, class_weight="balanced")

# Linear SVC
def lin_svm_classifier():
    return svm.LinearSVC(C=1, loss='hinge', penalty='l2', class_weight="balanced")

# Random Forest
def rf_classifier():
    #r = ensemble.RandomForestClassifier(class_weight='balanced')
    e = ensemble.ExtraTreesClassifier(random_state=0, class_weight='balanced')
    #b = ensemble.BaggingClassifier()
    return e

# KNN
def knn_classifier(k):
    return KNeighborsClassifier(n_neighbors=k)



# Import list of genes, return original sample set filtered to these genes only
def import_features(filename, headers, samples):

    # Map genes to their indices in the sample set for quick lookup
    gene_indices = {k: v for v, k in enumerate(headers)}

    with open(filename) as f:
        genes = [gene.strip('\n') for gene in f.readlines()]

    # Determine indices of genes read from file, return sample subset
    filtered_gene_indices = [gene_indices[gene] for gene in genes]
    return filter_cols(samples, filtered_gene_indices)


# Pase CSV, separate and format data
def split_data(filename):

    # Read headers, then data separately
    headers = list(pd.read_csv(filename, index_col=0,  nrows=1).columns)
    data = pd.read_csv(filename, header=0).as_matrix()

    # Remove 'Ti' samples
    def legit(a):
        return not a[0].startswith('Ti')

    bool_arr = np.array([ legit(row) for row in data ])
    filtered = data[bool_arr]

    # Split input CSV into parts
    # s[0]-2D array of labels , s[1]-2D array of sample data

    s = np.split(filtered, [1], axis=1)

    # Simplify all labels to their inherent groupings (eg. T1, T2, Ta)
    labels = np.reshape(s[0], np.size(s[0]))
    labels = [cancer[:2] for cancer in labels]

    counter = Counter(labels)
    print(counter)

    return headers, s[1], labels


# Filter 2d array to only selected columns
def filter_cols(l, cols):
    return l[:, cols]



def main():

    filename = "Bladder cancer gene expressions.csv"
    headers, samples, labels = split_data(filename)

    # Filter samples to top 100 features as selected by several algorithms
    samples_RF = import_features("rf_100_best.txt", headers, samples)
    samples_RFE = import_features("rfe_100_best.txt", headers, samples)
    samples_CHI2 = import_features("chi2_100_best.txt", headers, samples)


    # Compare SVM classification across different feature selected samples
    print("===============================================")
    print("Testing SVM Classifier")
    print("===============================================")

    print("SVM Classification on original dataset")
    calculate_efficiency(svm_classifier(), samples, labels)

    print("\nFeature Selection: Random Forest (Feature Importance)")
    calculate_efficiency(svm_classifier(), samples_RF, labels)

    print("\nFeature Selection: Recursive Feature Elimination (RFE)")
    calculate_efficiency(svm_classifier(), samples_RFE, labels)

    print("\nFeature Selection: Chi^2 Scoring Function")
    calculate_efficiency(svm_classifier(), samples_CHI2, labels)



    # Compare RF classification across different feature selected samples
    print("\n\n===============================================")
    print("Testing RF Classifier")
    print("===============================================")

    print("RF Classification on original dataset")
    calculate_efficiency(rf_classifier(), samples, labels)

    print("\nFeature Selection: Random Forest (Feature Importance)")
    calculate_efficiency(rf_classifier(), samples_RF, labels)

    print("\nFeature Selection: Recursive Feature Elimination (RFE)")
    calculate_efficiency(rf_classifier(), samples_RFE, labels)

    print("\nFeature Selection: Chi^2 Scoring Function")
    calculate_efficiency(rf_classifier(), samples_CHI2, labels)



    # Compare KNN classification across different feature selected samples
    print("\n\n===============================================")
    print("Testing KNN Classifier, K=20")
    print("===============================================")

    print("KNN Classification on original dataset")
    calculate_efficiency(knn_classifier(20), samples, labels)

    print("\nFeature Selection: Random Forest (Feature Importance)")
    calculate_efficiency(knn_classifier(20), samples_RF, labels)

    print("\nFeature Selection: Recursive Feature Elimination (RFE)")
    calculate_efficiency(knn_classifier(20), samples_RFE, labels)

    print("\nFeature Selection: Chi^2 Scoring Function")
    calculate_efficiency(knn_classifier(20), samples_CHI2, labels)

main()
