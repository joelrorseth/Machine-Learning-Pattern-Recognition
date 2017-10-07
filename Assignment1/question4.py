#
# Classifiers (10 Fold Cross Validation)
# 60-473 Assignment 1 Q4
#
# Determine the 'best' value of k for the K Nearest Neighbor classifier.
# Compare its efficiency with that of the 1-NN and Naive Bayes classifier.
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



# Determine the value of k that yeilds the most accurate KNN classifier
def find_best_k_knn(examples, labels):
    best_acc = 0.0
    best_k = -1
    temp = []

    for k in range(200):

        # Determine accuracy of KNN with k
        tn, fp, fn, tp = classify_knn(examples, labels, k+1)
        acc = accuracy(tn, fp, fn, tp)
        temp.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_k = k+1

    # Uncomment to see all accuracy readings sequentially for all k's
    #for v in temp:
    #    print("%.3f " % v, end=" ")
    #print()

    return best_k, best_acc



def classify_knn(examples, labels, k):

    # Fit training samples against sample labels
    neighbors = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    # Use confusion matrix to extract statistics on cross validated prediction
    prediction = cross_val_predict(neighbors, examples, labels, cv=10)
    return confusion_matrix(labels, prediction).ravel()



def classify_naive_bayes(examples, labels):

    bayes = GaussianNB()

    # Confusion matrix will extract Naive Bayes  prediction results
    prediction = cross_val_predict(bayes, examples, labels, cv=10)
    return confusion_matrix(labels, prediction).ravel()



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


    #input_files = ["twogaussians.csv"]

    for filename in input_files:
        examples, labels = split_data(filename)

        print("-------------------------------")
        print(filename, "\n-------------------------------")


        best_k, best_acc = find_best_k_knn(examples, labels)
        print("Best k for KNN classifier on", filename, "is", best_k, "with", best_acc)


        tn, fp, fn, tp = classify_knn(examples, labels, 1)
        print("The accuracy of KNN with k=1 is", accuracy(tn, fp, fn, tp))

        tn, fp, fn, tp = classify_naive_bayes(examples, labels)
        print("The accuracy of Naive Bayes is", accuracy(tn, fp, fn, tp))

        print()


main()
