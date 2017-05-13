from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

label_column = df_train.columns[0]

X_train = df_train.drop(label_column, axis=1)
Y_train = df_train[label_column]

clf = MLPClassifier(hidden_layer_sizes=(100,), solver="sgd", alpha=1e-4)
# clf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=2, min_samples_leaf=1)


clf.fit(X_train, Y_train)

print(Y_train)

predictions = clf.predict(df_test)


log = open("answer.csv", "w")
log.write("ImageId,Label\n")

for i in range(len(predictions)):
    log.write(",".join(map(str, [i+1, predictions[i]])) + "\n")

log.close()

