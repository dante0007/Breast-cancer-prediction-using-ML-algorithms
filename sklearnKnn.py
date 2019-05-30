import pandas as pd
import numpy as np
from sklearn import neighbors, preprocessing, model_selection

# read the csv file 
data = pd.read_csv('breast-cancer-wisconsin.data')

# unwanted id column
data.drop(['id'], 1, inplace=True)

# missing entries
data.replace('?', -9999, inplace=True)

# getting attributes
X = np.array(data.drop(['class'], 1))
y = np.array(data['class'])

# training and testing 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

# initialize classifier
knn = neighbors.KNeighborsClassifier()

# classifier training data
knn.fit(X_train, y_train)

# calculating accuracy 
accuracy = knn.score(X_test, y_test)

# prediction
new_tests = np.array([[10, 10, 2, 3, 10, 2, 1, 8, 44], [10, 1, 12, 3, 1, 12, 1, 8, 12], [3, 1, 1, 3, 1, 12, 1, 2, 1]])
new_tests = new_tests.reshape(len(new_tests), -1)
prediction = knn.predict(new_tests)

# output
print ("Accuracy: ", accuracy)

print ("Predictions:")
for pred in prediction:
	if pred == 2:
		print (pred, "Benign")
	else: print (pred, "Malignant")

