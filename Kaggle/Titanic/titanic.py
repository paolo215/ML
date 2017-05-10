import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier



train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
mean_age_train = np.mean(train_df["Age"])
mean_age_test = np.mean(test_df["Age"])
train_df["Age"] = train_df["Age"].fillna(mean_age_train)
test_df["Age"] = test_df["Age"].fillna(mean_age_test)
train_df["Embarked"] = train_df["Embarked"].fillna(-1)
test_df["Embarked"] = test_df["Embarked"].fillna(-1)

X_train = train_df.drop(["Name", "Ticket", "Cabin", "Survived", "PassengerId"], axis=1)

import math

X_train["Sex"][X_train["Sex"] == "female"] = 1
X_train["Sex"][X_train["Sex"] == "male"] = 0
X_train["Embarked"][X_train["Embarked"] == "Q"] = 0
X_train["Embarked"][X_train["Embarked"] == "S"] = 1
X_train["Embarked"][X_train["Embarked"] == "C"] = 2


#plt.hist(train_df["Age"], bins=np.arange(train_df["Age"].min(), train_df["Age"].max()))

plt.hist([train_df[train_df["Survived"]==1]["Age"], train_df[train_df["Survived"]==0]["Age"]], bins=30, stacked=True, color=["g", "r"] , label=["Survived", "Dead"])
plt.savefig("Age.png")
plt.clf()


Y_train = train_df["Survived"]


X_test = test_df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
X_test["Sex"][X_test["Sex"] == "female"] = 1
X_test["Sex"][X_test["Sex"] == "male"] = 0
X_test["Embarked"][X_test["Embarked"] == "Q"] = 0
X_test["Embarked"][X_test["Embarked"] == "S"] = 1
X_test["Embarked"][X_test["Embarked"] == "C"] = 2

X_test = X_test.fillna(-1)

test_ids = test_df["PassengerId"]

rfc = RandomForestClassifier(n_estimators=100, max_depth = 10, min_samples_split=2, random_state = 1)
rfc.fit(X_train, Y_train)
predictions = rfc.predict(X_test)



log = open("answers.csv", "w")
log.write("PassengerId,Survived\n")
for i in range(len(predictions)):
    log.write(",".join([str(test_ids[i]), str(predictions[i])]) + "\n")


log.close()
