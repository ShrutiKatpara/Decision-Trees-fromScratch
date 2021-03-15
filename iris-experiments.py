import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import csv

np.random.seed(42)

# Read IRIS data set
# ...
# 

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

def shuffle_dataset(X,y):
    X['y'] = y

    df = X.sample(frac=1, random_state=42)
    df.reset_index(drop=True,inplace=True)
    y = df.pop('y')
    return df,y

def train_and_predict(X,y, max_depth=15):
    """Function to train and predict iris using my decision tree"""
    clf = DecisionTree(criterion="information_gain",max_depth=max_depth)

    clf.fit(pd.DataFrame(X[0:120]),pd.Series(y[0:120],dtype = "category"))

    y = y[120:]

    y_hat = clf.predict(pd.DataFrame(X[120:]))

    print("Accuracy",accuracy(pd.Series(y_hat),pd.Series(y)))

    y = pd.Series(y)

    for cls in y.unique():
        print('Precision: ',cls," : ", precision(y_hat, y, cls))
        print('Recall: ',cls," : ", recall(y_hat, y, cls))

def five_fold_validation(X,y,depth=5):
    """Function to do five fold cross validation on iris"""
    X_original = X
    y_original = y

    accs = []

    # last 5th chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(X[0:120]),pd.Series(y[0:120],dtype = "category"))
    y_hat = clf.predict(pd.DataFrame(X[120:]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[120:])))

    # 4rd chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    pass_X = pd.DataFrame(np.append(X[90:],X[0:60],axis=0))
    pass_y = pd.Series(np.append(y[90:],y[0:60],axis=0), dtype="category")
    clf.fit(pass_X , pass_y)
    y_hat = clf.predict(pd.DataFrame(X[60:90]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[60:90])))
    
    # 3nd chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[120:],X[0:90],axis=0)), pd.Series(np.append(y[120:],y[0:90],axis=0),dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[90:120]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[90:120])))
    
    # 2st chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(X[30:]), pd.Series(y[30:],dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[0:30]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[0:30])))
    
    # 1st chunk as test data
    clf = DecisionTree(criterion="information_gain",max_depth=depth)
    clf.fit(pd.DataFrame(np.append(X[0:30],X[60:],axis=0)), pd.Series(np.append(y[0:30],y[60:],axis=0),dtype="category"))
    y_hat = clf.predict(pd.DataFrame(X[30:60]))
    accs.append(accuracy(pd.Series(y_hat),pd.Series(y[30:60])))

    print("Individual Accuracies:")
    print(*accs)
    print("Average Accuracy:")
    avg = sum(accs)/5
    print(avg)

def nested_validation(dataset,y,depth=5):
    for i in range(5):
        
        test = dataset[30*i:30*(i+1)]
        test_label = y[30*i:30*(i+1)]
        
        if 30*(i+1)+120<=150:
            train = dataset[30*(i+1):]
            train_label = y[30*(i+1):]
        else:
            train1 = dataset[0:30*(i+1)-30]
            train1_label = y[0:30*(i+1)-30]
            train2 = dataset[30*(i+1):]
            train2_label = y[30*(i+1):]
            train = np.append(train1,train2,axis=0)
            train_label = np.append(train1_label,train2_label,axis=0)
        accuracy_validation = {}
        for depth in range(1,11):
            avg_acc = 0
            for j in range(4):
                validation = train[30*j:30*(j+1)]
                validation_label = train_label[30*j:30*(j+1)]
                train_1 = train[30*(j+1):]
                train1_label = train_label[30*(j+1):]
                train_2 = train[0:30*(j+1)-30]
                train2_label = train_label[0:30*(j+1)-30]
                train_new= np.append(train_1,train_2,axis = 0)
                train_new_label = np.append(train1_label,train2_label,axis=0)
                tree = DecisionTree(criterion="gini_index",max_depth=depth)
                train_new=pd.DataFrame(train_new)
                train_new_label = pd.Series(train_new_label,dtype="category")
                train_new.reset_index(drop=True,inplace= True)
                train_new_label.reset_index(drop=True,inplace= True)
                tree.fit(train_new,train_new_label)
                avg_acc+= accuracy(tree.predict(validation),validation_label)
            accuracy_validation[depth] = avg_acc/4
        value = max(accuracy_validation, key = accuracy_validation.get)
        tree = DecisionTree(criterion="gini_index",max_depth=value)
        train = pd.DataFrame(train)
        train_label = pd.Series(train_label,dtype="category")

        tree.fit(train,train_label)
        print("Accuracy is,",accuracy(tree.predict(test),test_label), " for iteration",i+1, ". The depth of the optimal tree is ",value)

X,y = load_iris()

X,y = shuffle_dataset(pd.DataFrame(X), pd.Series(y))
# print(X,y)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n")
train_and_predict(X,y)
print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\n")
print("Five Fold Cross Validation")
five_fold_validation(X,y)
print("\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("\n")
# print("Nested Cross Validation")
# nested_validation(X,y)
# print("\n")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")