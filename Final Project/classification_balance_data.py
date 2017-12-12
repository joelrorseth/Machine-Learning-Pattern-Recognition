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
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from numpy import concatenate
import random

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

    #print("Average accuracy over CV runs:", np.average(score))
    print(np.average(score))
    # Print confusion matrix and other stats
    #print(classification_report(labels, pred))

    #conf_matrix = confusion_matrix(labels, pred)
    #print("Confusion Matrix:\n", conf_matrix)



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
    #print("RF: Finding", K, "best genes\n")

    forest_clf = rf_classifier()
    forest_clf.fit(samples, labels)

    importances = forest_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print top k features by their headers (gene)
    #headers_at_indices(headers, indices[:K])

    samples_RF = filter_cols(samples, indices[:K])
    print("RF: ", K)
    calculate_efficiency(forest_clf, samples_RF, labels)

# Use chi^2 statistical test via SelectKBest to find most important features
def k_best(K, headers, samples, labels):
    #print("Chi^2: Finding", K, "best genes\n")

    clf = rf_classifier()
    clf.fit(samples, labels)
    test = SelectKBest(score_func=chi2, k=K)
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
    #headers_at_indices(headers, indices)

    samples_Chi = filter_cols(samples, indices)
    print("CHI: ", K)
    calculate_efficiency(clf, samples_Chi, labels)
    #samples_to_show = 20
    #print(features[0:(samples_to_show),:])


# Use Recursive Feature Elimination (RFE) to determine important features
# TODO: This is taking forever, might not be feasible
def rfe_find_important(K, samples, labels):
    print("RFE: Finding", K, "best genes\n")

    model = LogisticRegression()
    print("[X] LR created")
    rfe = RFE(model, K)
    print("[X] RFE created")

    fitted = rfe.fit(samples, labels)
    print("[X] RFE Fit")

    print("Num Features: %d") % fitted.n_features_
    print("Selected Features: %s") % fitted.support_
    print("Feature Ranking: %s") % fitted.ranking_


# Print the gene names (header) of the specified indices
def headers_at_indices(headers, indices):

    print("Index \t Gene Biomarker")
    for i in sorted(indices):
        print(i, "\t", headers[i])
    print("\n\n")


# Return samples array modified to duplicate under-represented class' samples
def balance_samples_to_largest(samples, labels):
    balanced_s = samples
    balanced_l = labels

    # Determine class with the most samples belonging, scale others to this
    counter = Counter(labels)
    largest_rep = counter.most_common()[0][1]

    for stage in counter.keys():
        # For classes with less than largest representation, insert more
        if counter[stage] < largest_rep:

            amount = largest_rep - counter[stage]
            balanced_s, balanced_l = reinsert_samples(\
                    amount, stage, balanced_s, balanced_l)

    return balanced_s, balanced_l


# Insert 'amount' duplicated samples belonging to class 'stage' into 'samples'
def reinsert_samples(amount, stage, samples, labels):
    modified_s = samples
    modified_l = labels

    # Define filter function to obtain array of only samples of class 'stage'
    stage_mask = np.array([ (label == stage) for label in labels ])
    filtered_samples = samples[stage_mask]

    # Insert random samples back into samples (and label into labels)
    for i in range(amount):
        modified_s = np.concatenate((modified_s,\
                [random.choice(filtered_samples)]))
        modified_l.append(stage)

    return modified_s, modified_l


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

    #balance out T1 to Ta
    #def T1_stage_extracted(a):
    #    return a[0].startswith('T1')
    #bool_arrT1 = np.array([ T1_stage_extracted(row) for row in filtered ])
    #data_temp_T1 = filtered
    #T1_data = data_temp_T1[bool_arrT1]
    #filtered = concatenate((filtered,T1_data[:4]), axis=0)

    #balance out T2 to Ta
    #def T2_stage_extracted(a):
    #    return a[0].startswith('T2')
    #bool_arrT2 = np.array([ T2_stage_extracted(row) for row in filtered ])
    #data_temp_T2 = filtered
    #T2_data = data_temp_T2[bool_arrT2]
    #T2_data = concatenate((T2_data,T2_data,T2_data), axis=0)
    #filtered = concatenate((filtered,T2_data[:19]), axis=0)

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

    # Attempt manual class balancing
    balance_samples_to_largest(samples, labels)

    #filtered_samples = feature_select(samples, labels)

    # print("Sample size before:", samples.shape)
    # print("Sample size after:", filtered_samples.shape)


    # print("===============================================")
    # print("Testing SVM Classifier")
    # print("===============================================")
    # cl = svm_classifier(1)
    # calculate_efficiency(cl, filtered_samples, labels)
    # print()

    #print("\n===============================================")
    #print("Testing Forest w/ original samples")
    #print("===============================================")
    #calculate_efficiency(forest, samples, labels)


    # Test out differnt feature selection algorithms to find most important
    #for k in range(1, 100):
    #    k_most_important(k, headers, samples, labels)
    #    k_best(k, headers, samples, labels)
    #rfe_find_important(10, samples, labels)


main()
