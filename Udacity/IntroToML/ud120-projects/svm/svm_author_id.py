#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
#clf = SVC(kernel="linear")
#clf = SVC(kernel="rbf")

clf10 = SVC(kernel="rbf", C=10.0)
clf100 = SVC(kernel="rbf", C=100.0)
clf1000 = SVC(kernel="rbf", C=1000.0)
clf10000 = SVC(kernel="rbf", C=10000.0)


#features_train = features_train[:len(features_train) / 100]
#labels_train = labels_train[:len(labels_train) / 100]

#clf10.fit(features_train, labels_train)
#clf100.fit(features_train, labels_train)
#clf1000.fit(features_train, labels_train)
clf10000.fit(features_train, labels_train)


# Output accuracy
#print("C=10", clf10.score(features_test, labels_test))
#print("C=100", clf100.score(features_test, labels_test))
#print("C=1000", clf1000.score(features_test, labels_test))
#print("C=10000", clf10000.score(features_test, labels_test))

# Give predictions for element 10, 26, 50
pred10000 = clf10000.predict(features_test)
#print("10", pred10000[10])
#print("26", pred10000[26])
#print("50", pred10000[50])

# How many predicdted to be in 1 class
print(pred10000[pred10000 == 1].size)

#########################################################


