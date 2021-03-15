
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.random_Xs = list()
        self.random_ys = list()

        self.trees = list()

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.data = X
        self.labels = y
        for estimator in range(self.n_estimators):
            Dtree =  tree.DecisionTreeClassifier()
            # randomly make new dataset by with replcaement proces for samples
            X_train,y_train = self.random_split(X,y)
            # print(X_train, y_train)
            
            self.random_Xs.append(X_train)
            self.random_ys.append(y_train)
            
            Dtree.fit(X_train,y_train)
            
            self.trees.append(Dtree)

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

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        out = "y"
        if (isinstance(X, pd.DataFrame)):
            if (out in X.columns):
                X = X.drop(['y'],axis=1)
        
        y_hat = np.zeros(len(X))
        
        all_predictions = []
        for tree in self.trees:
            all_predictions.append(tree.predict(X))
        
        pred_arr = np.array(all_predictions)
        pred_arr = pred_arr.T
        # pred max pred value for each samples
        y_hat = [np.argmax(np.bincount(i)) for i in pred_arr]
        
        return(pd.Series(y_hat))

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        
        plot_colors = "mc"
        plot_step = 0.02
        n_classes = 2
        
        for est in range(self.n_estimators):

            plt.subplot(1,3, est+1)
            
            x_min, x_max = self.random_Xs[est].iloc[:, 0].min() - 1, self.random_Xs[est].iloc[:, 0].max() + 1
            y_min, y_max = self.random_Xs[est].iloc[:, 1].min() - 1, self.random_Xs[est].iloc[:, 1].max() + 1
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            
            Z = self.trees[est].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.random_ys[est] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.random_Xs[est].loc[idx[0][i]][0], self.random_Xs[est].loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        
        plt.suptitle("Decision Tree plots for individual estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        # plt.savefig("./figures/bagging_ind.png", dpi=400)
        fig1 = plt



        # Figure 2
        plot_colors = "gb"
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
        
        plt.suptitle("Decision Tree plot for combined estimator")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        # plt.savefig("./figures/bagging_comb.png", dpi=400)
        fig2 = plt

        return [fig1,fig2]
