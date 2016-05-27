import pandas as pd
import quandl #you can find kinds of datasets here
import math
import datetime
import time
import numpy as np
import matplotlib
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.style.use("ggplot")
plt.switch_backend("agg")


#data frame
df = quandl.get("WIKI/GOOGL")
#CHECK DOCS to see which algorithms that can be threaded

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

#print df.head()

forecast_col = "Adj. Close"
df.fillna(-99999, inplace=True) #replace NaN data
forecast_out = int(math.ceil(0.01*len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(["label"], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]


#scale X
df.dropna(inplace=True)
y = np.array(df["label"])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2) #take features, shuffle, and output 

clf = LinearRegression(n_jobs=10)
#clf = svm.SVR(kernel="poly")
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)


forecast_set = clf.predict(x_lately)

print forecast_set, accuracy, forecast_out

df["Forecast"] = np.nan
last_date = df.iloc[-1].name #last date
print last_date
last_unix = time.mktime(datetime.date(last_date.year, last_date.month, last_date.day).timetuple())
one_day = 86400
next_unix = last_unix + one_day

#Add dates to axis
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	#make future features NaN by iterating through the forecast
	#.loc referencing next_date in df
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] 


df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Pirce")
plt.show()










