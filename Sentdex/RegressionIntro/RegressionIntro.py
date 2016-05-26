import pandas as pd
import quandl #you can find kinds of datasets here
import math
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import LinearRegression
#data frame
df = quandl.get("WIKI/GOOGL")
#print df.head()

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

#print df.head()

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True) #replace NaN data
forecast_out = int(math.ceil(0.01*len(df)))

print forecast_out #days in advance

df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


x = np.array(df.drop(["label"], 1))
y = np.array(df["label"])
#scale X
x = preprocessing.scale(x)


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) #take features, shuffle, and output 

clf = LinearRegression(n_jobs=10)
#clf = svm.SVR(kernel="poly")
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print accuracy
#CHECK DOCS to see which algorithms that can be threaded


