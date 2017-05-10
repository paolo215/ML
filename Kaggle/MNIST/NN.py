from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

label_column = df_train.columns[len(df_train.columns) - 1]

X_train = df_train.drop(label_column, axis=1)
Y_train = df_train[label_column]

clf = MLPClassifier(hidden_layer_sizes=(25,), solver="lbfgs", alpha=1e-4)

clf.fit(X_train, Y_train)
predictions = clf.predict(df_test)

print(clf.score(X_train, Y_train))


log = open("answer.csv", "w")
log.write("ImageId,Label\n")

for i in range(len(predictions)):
    log.write(",".join(map(str, [i+1, predictions[i]])) + "\n")

log.close()

