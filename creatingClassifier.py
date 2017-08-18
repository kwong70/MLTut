import random 
from scipy.spatial import distance 

def euc(a,b):
	return distance.euclidean(a,b)

#We created our own classifier to guess randomly a flower testing set. 
class SimpleKNN():
	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train 

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(Y_train)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_dist= euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist<best_dist:
				best_dist = dist
				best_index = i
		return self.Y_train[best_index]

from sklearn import datasets
import numpy as np
from sklearn import  tree
iris = datasets.load_iris()

X = iris.data #Flower data aka features
Y = iris.target #Type of flower for each data aka target

from sklearn.cross_validation import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .5)

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier() #Creates nearest neighbor classifier 
#my_classifier = tree.DecisionTreeClassifier() #Creates  tree classifier 
my_classifier = SimpleKNN()

my_classifier.fit(X_train, Y_train) #Sending in features and corresponding targets (This teaches classifier kind of features makes a certain flower)
predicitons = my_classifier.predict(X_test) #Classifier knows how to classify certain flowers, so lets test it out with out testing data. Returns array with flower prediciton

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predicitons) #We got our predicitons now we're seeing of they were right is Y_test since that rep our answers

