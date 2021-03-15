import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

import csv

###Write code here

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

criteria = 'information_gain'

Classifier_AB = RandomForestClassifier(10, criterion = criteria)
Classifier_AB.fit(pd.DataFrame(X[0:90]), pd.Series(y[0:90]))

y_hat = Classifier_AB.predict(pd.DataFrame(X[90:150]))

print("Accuracy:", accuracy(y_hat,y[90:150]))

for cls in y.unique():
    print('Precision for',cls,' : ', precision(y_hat, y[90:150], cls))
    print('Recall for ',cls ,': ', recall(y_hat, y[90:150], cls))