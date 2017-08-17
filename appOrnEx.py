from sklearn import tree

features = [[140, 1], [130,1], [150, 0], [170,0]] # Weight, bumpy or not
labels = [0 , 0, 1, 1] # 0 = apples, 1 = oranges. These are the answers to the features
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels) #Training classifier with already given answers 

print clf.predict([150,0]) #Tests the classifier 

