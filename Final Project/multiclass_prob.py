#
# Bladder Cancer Subtypes Classification
# 60-473 Final Project
#
# Authors:
# - Joel Rorseth
# - Michael Bianchi
#
# This file explores potential solutions to The Multi-Class Problem, aiming
# to test multiclass algorithms that fall into the cateogyr of One-Vs-One,
# One-Vs-All and inherently multiclass classifiers.
#

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.cross_validation import StratifiedShuffleSplit


from collections import Counter
import numpy as np
import pandas as pd
import random

# Run tests and output stats for given classification
def calculate_efficiency(classifier, samples, labels):
    
    # Separate data into test and train sets
    samples_train, samples_test, labels_train, labels_test =\
        train_test_split(samples, labels, random_state=42, test_size=0.20, stratify=labels)
    
    print("Before:", Counter(labels_train))
    
    # Now we can avoid bias and safely rebalance classes by increasing minorities
    # This must be done in training set only! Otherwise dups could end up in train & test
    samples_train, labels_train =\
        balance_samples_to_largest(samples_train, labels_train)

    print("After:", Counter(labels_train))
    
    # Train samples on rebalanced sample set
    classifier.fit(samples_train, labels_train)
    
    print("Accuracy:", classifier.score(samples_test, labels_test))

    # Determine confusion matrix
    labels_pred = classifier.predict(samples_test)
    conf_matrix = confusion_matrix(labels_test, labels_pred)
    print(conf_matrix)

    # OLD CODE
    # Determine cross-validated score
    #pred = cross_val_predict(classifier, samples, labels, cv=5)
    #score = cross_val_score(classifier, samples, labels, cv=5,scoring="accuracy")
    #print("Average accuracy over CV runs:", np.average(score))
    # Print confusion matrix and other stats
    #print(classification_report(labels, pred))
    #conf_matrix = confusion_matrix(labels, pred)
    #print("Confusion Matrix:\n", conf_matrix)



# MARK: Multiclass Meta-Estimators
# One-Vs-All (aka One-Vs-Rest)
def one_vs_all(classifier):
    return OneVsRestClassifier(classifier)

# One-Vs-One
def one_vs_one(classifier):
    return OneVsOneClassifier(classifier)



# MARK: Base Classifiers
# SVM Classifier
def svm_classifier():
    return SVC(kernel='rbf', C=1, gamma='auto', class_weight='balanced')

# Linear SVC
def lin_svm_classifier():
    return LinearSVC(C=1, loss='hinge', penalty='l2',\
        class_weight="balanced")

# Nu-Support Vector Classification
def nu_svm_classifier():
    return NuSVC(kernel='rbf', gamma='auto', class_weight='balanced')


# One vs All Gaussian Process Classifier
def ova_gaussian_process():
    return GaussianProcessClassifier(multi_class="one_vs_rest")

# One vs One Gaussian Process Classifier
def ovo_gaussian_process():
    return GaussianProcessClassifier(multi_class="one_vs_one")


# Extra Trees Classifier
def extra_trees_classifier():
    return ExtraTreesClassifier(random_state=0, class_weight='balanced')

# Bagging Tree Classifier
def bagging_classifier():
    return BaggingClassifier()

# Random Forest Classifier
def rf_classifier():
    return RandomForestClassifier(class_weight='balanced')

# Gradient Boosting
def gradient_boosting():
    return GradientBoostingClassifier()


def logistic_regression():
    return LogisticRegression(multi_class="ovr")

def sgd_classifier():
    return SGDClassifier()

def perceptron():
    return Perceptron()


# KNN w/ k=20
def knn_classifier():
    return KNeighborsClassifier(n_neighbors=20)


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

    
    for k in range(1,11):
        # Filter samples to top 100 features as selected by several algorithms
        samples_RF = import_features("rf_" + str(k) + "_best.txt", headers, samples)
        
        classifiers = [svm_classifier, lin_svm_classifier, rf_classifier,\
                extra_trees_classifier, bagging_classifier, knn_classifier,\
                ova_gaussian_process, ovo_gaussian_process,\
                logistic_regression, sgd_classifier, perceptron]

        names = ["SVM Classifier RBF Kernel", "SVM Classifier Linear Kernel",\
                "Random Forest Classifier", "Extra Trees Classifier", \
                "Bagging Classifier", "K-Nearest Neighbors Classifier K=20",
                "One-Vs.-All Gaussian Process",\
                "One-Vs.-One Gaussian Process", "Logistic Regression",\
                "Perceptron"]

        # Test each classifier on its own, One vs. All and One vs. One
        for i in range(len(classifiers)):
            classifier = classifiers[i]
            name = names[i-1]

            print("===============================================")
            print("Testing", name, "kbest: ", k)
            print("===============================================")

            print("Inherent Multi-Class Classification:")
            calculate_efficiency(classifier(), samples_RF, labels)

            print("Used in One-Vs-All Meta-Estimator:")
            o_vs_a = one_vs_all(classifier())
            calculate_efficiency(o_vs_a, samples_RF, labels)

            print("Used in One-Vs-One Meta-Estimator:")
            o_vs_o = one_vs_one(classifier())
            calculate_efficiency(o_vs_o, samples_RF, labels)

            print()

        print()
        print()
        print()
        print()
        print()
        print()
        print()
        print("Next run")
        print()
        print()
        print()
        print()
        print()
        print()
        print()


main()
