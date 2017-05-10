import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("iris.data", header=None)
df.fillna(0, inplace=True)

dfTrain, dfTest = train_test_split(df, test_size=0.5)

clf = DecisionTreeClassifier()

labelCol = len(df.columns) - 1

X_train = dfTrain.drop(labelCol, 1)
Y_train = dfTrain[labelCol]


X_test = dfTest.drop(labelCol, 1)
Y_test = dfTest[labelCol]


clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

print(clf.score(X_test, Y_test))

export_graphviz(clf, out_file="tree.dot")



