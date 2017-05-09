from sklearn.linear_model  import LinearRegression
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fileObj = open("laptop_battery_life.csv")
data = [map(float, f.split(",")) for f in fileObj.read().split("\n") if f]
data = np.array(data)
fileObj.close()


X_train = data[:,0].reshape(len(data), 1)
Y_train = data[:,1].reshape(len(data), 1)


"""
# Only scored 68% when testing training set
clf = LinearRegression()
clf.fit(X_train, Y_train)
print(clf.score(X_train, Y_train))
"""

plt.scatter(X_train, Y_train)
plt.savefig("laptop.png")
plt.clf()



correct = 0
for i in range(len(X_train)):
    value = X_train[i] * 2
    if value > 8.0:
        value = 8.0
    if Y_train[i] == value:
        correct += 1

print(correct / float(len(data)))


