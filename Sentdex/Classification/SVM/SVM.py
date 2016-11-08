import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import svm
import pandas as pd

df = pd.read_csv("breast-cancer-wisconsin.data")

# So it recognizes as an outlier
df.replace("?", -99999, inplace=True)

# Drop irrelevant
df.drop(["id"], 1, inplace=True)

# X = features
# y = label
X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)

print accuracy














