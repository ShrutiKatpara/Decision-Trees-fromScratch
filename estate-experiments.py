
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Read real-estate data set
# ...
# 

def load_estate():
    """Load the estate dataset from csv file"""
    X = []
    y = []
    with open('./dataset/estate.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count=0
        for row in csv_reader:
            row_count+=1
            if (row_count==1):
                continue
            l = len(row)
            temp = [float(row[i]) for i in range(1,l)]
            # print(temp)
            X.append(temp[:len(temp)-1])
            y.append(temp[-1])
    return X,y

def sktrain(X,y,max_depth=5,criterion='mse'):
    """Function to train and predict on estate dataset using sklearn decision tree"""
    # Dropping any rows with Nan values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

    regressor = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth) 

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    score = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rscore = math.sqrt(score)

    return rscore

def my_regr(X,y,max_depth=5,criterion="information_gain"):
    """Function to train and predict on estate dataset using my decision tree"""
    clf = DecisionTree(criterion=criterion,max_depth=max_depth)

    clf.fit(pd.DataFrame(X[0:330]),pd.Series(y[0:330]))

    # clf.plot()

    y = y[330:]

    y_hat = clf.predict(pd.DataFrame(X[330:]))

    y = pd.Series(y)

    print(rmse(y_hat,y))
    print(mae(y_hat,y))

X,y = load_estate()

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

print("\n")

print("\n")
print("1) using sklearn: ")
print("using RMSE score and MAE score respectively")
print(sktrain(X,y))
print(sktrain(X,y,criterion='mae'))

print("\n\n")

print("2) using my decision tree: ")
print("using RMSE score and MAE score respectively")
my_regr(X,y)

print("\n")

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")