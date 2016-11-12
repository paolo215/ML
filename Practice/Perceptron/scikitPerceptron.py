import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import cross_validation

from sklearn.linear_model import Perceptron

import string

data = {}

df = pd.read_csv("letter-recognition.data")
label = df["lettr"]

bias = [1 for _ in range(len(df))]
df.insert(1, "bias", bias)


for index, row in df.iterrows():
	if row["lettr"] not in data:
		data[row["lettr"]] = []
	newRow = np.array(row[1:], dtype=np.float64)
	data[row["lettr"]].append(newRow)

letters = list(string.ascii_uppercase)
votes = 0
total = 0
for a in letters:
	compareWith = list(letters)
	compareWith.remove(a)
	for b in compareWith:
		combined = data[a] + data[b]
		letter = [a] * len(data[a])
		letter += [b] * len(data[b])


		X_train, X_test, letter_train, letter_test = cross_validation.train_test_split(combined, letter, test_size=0.5)


		clf = Perceptron()
		clf.fit(X_train, letter_train)
		#accuracy = clf.score(X_test, letter_test)
		#print accuracy

		predictions = clf.predict(X_test)
		for idx, prediction in enumerate(predictions):
			if prediction == letter_test[idx]:
				votes += 1
			total += 1

		print a + " == " + b, votes, total, float(votes) / total


print votes, total, float(votes) / total

#X = np.array(df.drop(["lettr"], 1))
#X = preprocessing.scale(X)

#X_train, X_test, letter_train, letter_test = cross_validation.train_test_split(X, letter, test_size=0.5)
#clf = Perceptron()
#clf.fit(X_train, letter_train)
#accuracy = clf.score(X_test, letter_test)

#prediction = clf.predict(example)



