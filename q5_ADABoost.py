"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn import tree
from sklearn.metrics import precision_score, recall_score
import csv

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
Dtree =  tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=Dtree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

def load_iris():
    X = []
    y = []
    with open('./dataset/iris.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            l = len(row)
            temp = [float(row[i]) for i in range(l-1)]
            # print(temp)
            X.append(temp[:l-1])
            y.append(row[l-1])
    return X,y


# load the feature matrix and output vec from the dataset
xl,yl = load_iris()

X = pd.DataFrame(xl)
y = pd.Series(yl)
X['y'] = y

df = X.sample(frac=1, random_state=42)
df.reset_index(drop=True,inplace=True)
y = df.pop('y')

# dropping features except sepal width and petal widht
X = df.drop(columns=[0,2])
X = X.rename({1:0,3:1},axis=1)

n_estimators = 3
Dtree =  tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)

Classifier_AB = AdaBoostClassifier(base_estimator=Dtree, n_estimators=n_estimators )
Classifier_AB.fit(pd.DataFrame(X[0:90]), pd.Series(y[0:90]))

y_hat = Classifier_AB.predict(pd.DataFrame(X[90:150]))

print("Accuracy:", accuracy(y_hat,y[90:150]))

for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y[90:150], cls))
    print('Recall for ',cls ,': ', recall(y_hat, y[90:150], cls))