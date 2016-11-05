import quandl
import pandas as pd
import math
import time
import datetime
import numpy as np

# Use to scale data. Goal is to be between -1 to 1 to help accuracy
from sklearn import preprocessing

# shuffle and split up data so no biased sample
from sklearn import cross_validation

# support vector machine to do regression
from sklearn import svm

from sklearn.linear_model import LinearRegression


# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import style


import pickle


style.use("ggplot")

# df = data frame
df = quandl.get("WIKI/GOOGL")

# print df.head() # each column is a feature
# Simplify data as much as possible

# Get relevant dataframe
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df ["Adj. Close"] * 100.0


df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df ["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]


# Features are the attributes that make up the label
# Label is a prediction into the future

forecast_col = "Adj. Close"

# fill not available
# ML can't work with NaN data so replace or get rid 
# of data.
df.fillna("-99999", inplace=True)

# Regression Algorithm
# Get the number of days out (predict 10% of the df)
forecast_out = int(math.ceil(0.01*len(df)))


# Shifting col. negatively (up)
# Each row would be the Adj. Close price 10 days
# into the future.
df["label"] = df[forecast_col].shift(-forecast_out)

# print(df.head())
# Features = X
# Labels = y

X = np.array(df.drop(["label"], 1))

# Normalize all data
X = preprocessing.scale(X)

# X_lately are the stuff that we are going to predict
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df["label"])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# How easy to switch algorith
# clf = LinearRegression()
# clf = svm.SVR()
# Check documentation if LinearRegression can be threaded
# clf = LinearRegression(n_jobs=-1)


# train
# clf.fit(X_train, y_train)

# Save classifier
# with open("linearRegression.pickle", "wb") as f:
#	 pickle.dump(clf, f)


# Now we don't have to train the model every time we run this
pickle_in = open("linearRegression.pickle", "rb")
clf = pickle.load(pickle_in)



# test
accuracy = clf.score(X_test, y_test)

# You can pass in a value or an array
forecast_set = clf.predict(X_lately)

print forecast_set, accuracy, forecast_out

# Specify this column to be full of nan
df["Forecast"] = np.nan 

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	#iterate each forecast and day and set values in date frame to make future features nan
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Visualize data using matplot
# df["Adj. Close"].plot()
# df["Forecast"].plot()
# plt.legend(loc=4)
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.show()


print df.tail()














