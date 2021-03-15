"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

import time

np.random.seed(42)


""" Case: RIRO"""

learning_time = dict()
predict_time = dict()

for step in range(1,31):
    N = step*6
    P = step
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
        
    start_time = time.time()
    tree = DecisionTree(criterion="information_gain")
    tree.fit(X, y)
    end_time = time.time()
        
    learning_time[step] = (end_time-start_time)

    start_time = time.time()
    y_hat = tree.predict(X)
    end_time = time.time()
        
    predict_time[step] = (end_time-start_time)

fp = open("./dataset/riro.txt", 'w')
fp.write('learning')
fp.write("\n")
for key,value in learning_time.items():
    fp.write(str(key*6)+","+str(value))
    fp.write("\n")

fp.write('predict')
fp.write("\n")
for key,value in predict_time.items():
    fp.write(str(key*6)+","+str(value*1000))
    fp.write("\n")

fp.close()

""" Case: RIDO"""

learning_time = dict()
predict_time = dict()

for step in range(1,31):
    N = step*6
    P = step
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(P, size = N), dtype="category")
        
    start_time = time.time()
    tree = DecisionTree(criterion="information_gain")
    tree.fit(X, y)
    end_time = time.time()
        
    learning_time[step] = (end_time-start_time)

    start_time = time.time()
    y_hat = tree.predict(X)
    end_time = time.time()
        
    predict_time[step] = (end_time-start_time)

fp = open("./dataset/rido.txt", 'w')
fp.write('learning')
fp.write("\n")
for key,value in learning_time.items():
    fp.write(str(key*6)+","+str(value))
    fp.write("\n")

fp.write('predict')
fp.write("\n")
for key,value in predict_time.items():
    fp.write(str(key*6)+","+str(value*1000))
    fp.write("\n")

fp.close()

""" Case: DIRO"""

learning_time = dict()
predict_time = dict()

for step in range(1,31):
    N = step*6
    P = step
    X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randn(N))
        
    start_time = time.time()
    tree = DecisionTree(criterion="information_gain")
    tree.fit(X, y)
    end_time = time.time()
        
    learning_time[step] = (end_time-start_time)

    start_time = time.time()
    y_hat = tree.predict(X)
    end_time = time.time()
        
    predict_time[step] = (end_time-start_time)


fp = open("./dataset/diro.txt", 'w')
fp.write('learning')
fp.write("\n")
for key,value in learning_time.items():
    fp.write(str(key*6)+","+str(value))
    fp.write("\n")

fp.write('predict')
fp.write("\n")
for key,value in predict_time.items():
    fp.write(str(key*6)+","+str(value*1000))
    fp.write("\n")
fp.close()

""" Case: DIDO"""

learning_time = dict()
predict_time = dict()

for step in range(1,31):
    N = step*6
    P = step
    X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randint(P, size = N) , dtype="category")
        
    start_time = time.time()
    tree = DecisionTree(criterion="information_gain")
    tree.fit(X, y)
    end_time = time.time()
        
    learning_time[step] = (end_time-start_time)

    start_time = time.time()
    y_hat = tree.predict(X)
    end_time = time.time()
        
    predict_time[step] = (end_time-start_time)


fp = open("./dataset/dido.txt", 'w')
fp.write('learning')
fp.write("\n")
for key,value in learning_time.items():
    fp.write(str(key*6)+","+str(value))
    fp.write("\n")

fp.write('predict')
fp.write("\n")
for key,value in predict_time.items():
    fp.write(str(key*6)+","+str(value*1000))
    fp.write("\n")

fp.close()