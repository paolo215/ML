# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
from sklearn.linear_model import LinearRegression

datasets_info = raw_input()
features, rows = datasets_info.split( )
data = []
for i in range(int(rows)):
    line = map(float, raw_input().split( ))
    data.append(line)
    
data = np.array(data)
X_train = data[:,range(features)]
Y_train = data[:,len(features)]


predictions = int(raw_input())
tests = []
for i in range(int(predictions)):
    line = map(float, raw_input().split( ))
    tests.append(line)
    
tests = np.array(tests)



clf = LinearRegression()

clf.fit(X_train, Y_train)
predictions = clf.predict(tests)
for prediction in predictions:
    print(prediction)



