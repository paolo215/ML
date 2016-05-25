import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
iris = load_iris()

#print iris.feature_names
#print iris.target_names
#print iris.data[0] #features and examples
#print iris.target[0] #labels

#for i in range(len(iris.target)):
#    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

test_idx = [0, 50, 100]

#training data
#contains all data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)


# testing data
#contains all the removed ones
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


clf = tree.DecisionTreeClassifier()
#train it on our training data
clf.fit(train_data, train_target)


#print test_target
#print clf.predict(test_data)


#viz code
dot_data = StringIO()
tree.export_graphviz(clf, 
                    out_file=dot_data,
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    filled=True, rounded=True,
                    impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("ouput.pdf")

print test_data[0], test_target[0]
print iris.feature_names, iris.target_names

