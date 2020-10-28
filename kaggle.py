import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('final.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:, -1].values


#y = dataset.iloc[:, -1].values
dataset = pd.read_csv('Test.csv')
X1 = dataset.iloc[:,0:13].values

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)



# Predicting the Test set results
y_pred = classifier.predict(X1)

my_submission = pd.DataFrame({'Id': X1[:,0].astype(int), 'SalePrice': y_pred})
#my_submission = pd.DataFrame({'ID': X2, 'Target': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

print(X1[:,0])