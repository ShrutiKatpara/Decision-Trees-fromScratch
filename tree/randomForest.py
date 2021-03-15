from .base import DecisionTree
import random

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

def random_split(X,y):

    total_attr = (len(list(X.columns)))

    drop_attr = []
    for i in range(int(total_attr/2)):
        drop_attr.append(random.randint(0,total_attr-1))

    new_X = X.drop(drop_attr, axis=1)

    # print(new_X)

    new_X['out'] = y

    lst = []
    for i in range(len(new_X)):
        smpl = new_X.sample(n=1)
        lst.append(smpl)
    
    new_X = pd.concat(lst)
    new_X.reset_index(drop=True,inplace = True)

    new_y = new_X.pop('out')

    return new_X,new_y


class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        if criterion == 'gini':
            self.criterion = criterion
        else:
            self.criterion = 'entropy'

        self.n_estimators = n_estimators
        
        self.estimators_list = []
        
        self.split_x = []
        self.split_y = []
        
        self.data = None
        self.labels = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.data = X
        self.labels = y
        
        for estimator in range(self.n_estimators):
            
            #Dtree = DecisionTree(criterion=self.criterion)
            new_X, new_y = self.random_split(X,y)
            # print(new_X, new_y)
            
            Dtree =  DecisionTreeClassifier(max_depth = 4,criterion = self.criterion)

            self.split_x.append(new_X)
            self.split_y.append(new_y)

            Dtree.fit(new_X,new_y)

            self.estimators_list.append(Dtree)

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        out = "y"
        if (isinstance(X, pd.DataFrame)):
            if (out in X.columns):
                X = X.drop(['y'],axis=1)

        y_hat = []
        predictions = []
        for i in self.estimators_list:
            temp = i.predict(X)
            predictions.append(temp)

        pred_arr = np.array(predictions)
        pred_arr = pred_arr.T

        y_hat = [np.unique(i)[np.argmax(np.unique(i, return_counts=True)[1])] for i in pred_arr]
        return(pd.Series(y_hat))
        

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """

        plot_colors = "rg"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(2, 5, _+1 )
            x_min, x_max = self.split_x[_].iloc[:, 0].min() - 1, self.split_x[_].iloc[:, 0].max() + 1
            y_min, y_max = self.split_x[_].iloc[:, 1].min() - 1, self.split_x[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            print(xx,yy)
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.estimators_list[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.split_y[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.split_x[_].loc[idx[0][i]][0], self.split_x[_].loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("RandomForestClassifier:Decision surface of a decision tree using two features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig1 = plt

        # Figure 2
        print("Printing combining decision surface ")
        plot_colors = "rg"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.data.iloc[:, 0].min() - 1, self.data.iloc[:, 0].max() + 1
        y_min, y_max = self.data.iloc[:, 1].min() - 1, self.data.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(self.labels == i)
            for i in range (len(idx[0])):
                plt.scatter(self.data.loc[idx[0][i]][0], self.data.loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("RandomForestClassifier:Decision surface by combining all the estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig2 = plt

        return [fig1,fig2]

    def random_split(self,X_t,y_t):
        """
        Funtion to make a new dataset with replacement on original dataset
            Input:
            X: pd.DataFrame with rows as samples and columns as features
            Output:
            y: pd.Series with rows corresponding to output variable.
        """
        X_t['y'] = y_t
        lst = []
        for i in range(len(X_t)):
            smpl = X_t.sample(n=1)
            lst.append(smpl)
        X_t = pd.concat(lst)
        X_t.reset_index(drop=True,inplace = True)
        y_t = X_t.pop('y')
        return X_t,y_t

class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        if criterion == 'variance':
            self.criterion = 'mse'
        else:
            self.criterion = criterion
        self.n_estimators = n_estimators
        self.estimators_list = []
        self.split_x = []
        self.split_y = []
        self.data = None
        self.labels = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.data= X
        self.y = y
        for estimator in range(self.n_estimators):
            print("--------------------------------------", estimator, "----------------------------------------------")
            Dtree =  DecisionTree(criterion="information_gain")#DecisionTreeRegressor(max_depth = 4, criterion = self.criterion,max_features='auto')
            # randomly split dataset with replacment for features and samples
            X_train,y_train = random_split(X,y)
            # print(X_train, y_train)
            self.split_x.append(X_train)
            self.split_y.append(y_train)
            Dtree.fit(X_train,y_train)
            self.estimators_list.append(Dtree)
            # Dtree.plot()

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = np.zeros(len(X))
        predictions = []
        for i in self.estimators_list:
            predictions.append(i.predict(pd.DataFrame(X)))
        pred_arr = np.array(predictions)
        pred_arr = pred_arr.T
        # print(pred_arr)
        # pred the mean as pred value for overall tree
        y_hat = [np.mean(i) for i in pred_arr]
        # print(y_hat)
        return(pd.Series(y_hat))

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        
        plot_colors = "rg"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            
            plt.subplot(2, 5, _+1 )
            
            x_min, x_max = self.split_x[_].iloc[:, 0].min() - 1, self.split_x[_].iloc[:, 0].max() + 1
            y_min, y_max = self.split_x[_].iloc[:, 1].min() - 1, self.split_x[_].iloc[:, 1].max() + 1
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            
            Z = self.estimators_list[_].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            
            for i, color in zip(range(n_classes), plot_colors):
                # idx = np.where(self.split_y[_] == i)
                plt.scatter(self.split_x[_][0], self.split_x[_][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        
        plt.suptitle("RandomForestRegressor: Decision surface of a decision tree using two features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig1 = plt

        # Figure 2
        print("Printing decision surface by combining the individual estimators")
        plot_colors = "rg"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.data.iloc[:, 0].min() - 1, self.data.iloc[:, 0].max() + 1
        y_min, y_max = self.data.iloc[:, 1].min() - 1, self.data.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
        for i, color in zip(range(n_classes), plot_colors):
            # idx = np.where(self.labels == i)
            plt.scatter(self.data[0], self.data[1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("RandomForestRegressor: Decision surface by combining all the estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig2 = plt

        return [fig1,fig2]
