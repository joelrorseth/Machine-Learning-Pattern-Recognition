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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm, tree, ensemble, preprocessing

from sklearn.feature_selection import SelectPercentile, mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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
def svm_classifier(c):
    return svm.SVC(kernel='rbf', C=c, gamma=1, class_weight="balanced")

# Linear SVC
def lin_svm_classifier(c):
    return svm.LinearSVC(C=c, loss='hinge', penalty='l2', class_weight="balanced")

# Random Forest
def rf_classifier():
    #r = ensemble.RandomForestClassifier(class_weight='balanced')
    e = ensemble.ExtraTreesClassifier(random_state=0, class_weight='balanced')
    #b = ensemble.BaggingClassifier()
    return e

# KNN
def knn_classifier(k):
    return KNeighborsClassifier(n_neighbors=k)



# Process and filter features
def feature_select(samples, labels):

    # Take a percentage of the best features
    #s = SelectPercentile(chi2).fit_transform(samples, labels)

    # Remove features w/ low variance
    selector = VarianceThreshold(0.1)
    s = selector.fit_transform(samples)

    # Standardize dataset
    #scaled = (preprocessing.MinMaxScaler()).fit_transform(s)
    #return scaled

    # Remove columns with many 0's
    #bad_feature_idx = []
    #
    #for idx, feature_col in enumerate(s.T):
    #    if (np.sum(feature_col) >= 200):
    #        bad_feature_idx.append(idx)
    #return filter_cols(samples, bad_feature_idx)

    return s

# Determine the most important features to the Random Forest classification
def k_most_important(K, headers, samples, labels):
    print("RF: Finding", K, "best genes\n")

    forest_clf = rf_classifier()
    forest_clf.fit(samples, labels)

    importances = forest_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print top k features by their headers (gene)
    return headers_at_indices(headers, indices[:K])


# Use chi^2 statistical test via SelectKBest to find most important features
def k_best(K, score_fn, headers, samples, labels):
    print("Chi^2: Finding", K, "best genes\n")

    test = SelectKBest(score_func=score_fn, k=K)
    fitted = test.fit(samples, labels)

    # Summarize scores
    np.set_printoptions(precision=3)
    #print(fitted.scores_)

    # Get (58, K) 2D array where samples contain array of values for each
    # selected feature
    features = fitted.transform(samples)

    # Extract selected (best) indices and determine corresponding feature gene
    best_mask = fitted.get_support()
    indices = [idx for idx, is_best in enumerate(best_mask) if is_best == True]
    return headers_at_indices(headers, indices)

    #samples_to_show = 20
    #print(features[0:(samples_to_show),:])


def pca_k_best(K, headers, samples, labels):
    print("PCA: Finding", K, "best genes\n")

    pca = PCA(n_components=K)
    fitted = pca.fit(samples)

    print(fitted.components_.shape)
    print(fitted.components_)


# Use Recursive Feature Elimination (RFE) to determine important features
# TODO: This is taking forever, might not be feasible
def rfe_find_important(K, samples, labels):
    print("RFE: Finding", K, "best genes\n")

    model = LogisticRegression()
    rfe = RFE(model, K)

    fitted = rfe.fit(samples, labels)

    # Get mask for indicies that are in the top K
    best_mask = fitted.support_
    #print("Feature Ranking: %s") % fitted.ranking_

    indices = [idx for idx, is_best in enumerate(best_mask) if is_best == True]
    return headers_at_indices(headers, indices)


# Print the gene names (header) of the specified indices
def headers_at_indices(headers, indices):
    h = []

    print("Index \t Gene Biomarker")
    for i in indices:
        print(i, "\t", headers[i])
        h.append(str(headers[i]))

    print("\n\n")
    return indices, h


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

# Write string to file
def write_file(filename, data):
    with open(filename, "w") as output:
        for element in data:
            output.write(str(element) + '\n')


def main():

    filename = "Bladder cancer gene expressions.csv"

    headers, samples, labels = split_data(filename)
    filtered_samples = feature_select(samples, labels)

    print("Sample size before:", samples.shape)
    print("Sample size after:", filtered_samples.shape)


    print("===============================================")
    print("Testing SVM Classifier")
    print("===============================================")
    cl = svm_classifier(1)
    calculate_efficiency(cl, filtered_samples, labels)
    print()

    #print("\n===============================================")
    #print("Testing Forest w/ original samples")
    #print("===============================================")
    #calculate_efficiency(forest, samples, labels)

    #h_best = []
    #ind_best = []

    # Test out differnt feature selection algorithms to find most important
    ind_best, h_best = k_most_important(100, headers, samples, labels)
    #h_best += h_rf
    #ind_best += ind_rf
    write_file("rf_50_best.txt", h_best)

    # Can use information gain via f_classif or chi2
    ind_chi, h_chi = k_best(100, chi2, headers, samples, labels)
    h_best += h_chi
    ind_best += ind_chi
    write_file("chi2_50_best.txt", h_chi)

    # Indicies of most common -- not really working
    #common_idx = [i for i, _ in (Counter(ind)).most_common(50)]

    #pca_k_best(10, headers, samples, labels)

    # Write RFE results to file
    ind_rfe, h_rfe = rfe_find_important(100, samples, labels)
    write_file("rfe_50_best.txt", h_rfe)
    h_best += h_rfe
    ind_best += ind_rfe

    # Now that all results have been combined, print out 50 most common
    # among all evaluation methods

    h_common = (Counter(h_best)).most_common(50)
    #ind_common = (Counter(ind_best)).most_common(50)
    print(h_common)
    write_file("most_common_50_best.txt", h_common)

main()
