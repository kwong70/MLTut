from sklearn.datasets import load_iris 
import numpy as np
from sklearn import  tree

iris = load_iris()
print iris.feature_names
print iris.target_names

print iris.data[0]
test_idx = [0,50,100] #Removing the first of each type of flower to use as testing data 

#Training Data 
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print clf.predict(test_data)


