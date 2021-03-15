import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import precision_score, recall_score

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.estimators_list = list()

        self.all_Xs = list()
        self.all_ys = list()

        self.all_amount_of_says = list()


    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.out_classes = list(set(list(y)))
        self.data = X
        self.labels = y

        for estimator in range(self.n_estimators):
            print("--------------------------------------", estimator, "----------------------------------------------")

            self.all_Xs.append(X)
            self.all_ys.append(y)

            total_samples = len(X)
            # print("total_samples:", total_samples)

            sample_weights = [1/total_samples]*total_samples


            Dtree =  tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
            # fit the estimator
            Dtree.fit(X,y,sample_weight=sample_weights)
            
            self.estimators_list.append(Dtree)

            y_hat = Dtree.predict(X)

            # count all the wrong predicted output for curr estimator
            wrong_pred = 0
            index_wrong_pred = []
            for i in range(len(y)):
                if (y_hat[i]!=y[i]):
                    wrong_pred+=sample_weights[i]
                    index_wrong_pred.append(i)
            
            # add some delta value to prevent zero division err
            err = 0.00000001
            wrong_pred += err

            # calculate amount of say 
            amount_of_say = 0.5 * (math.log2(((1-wrong_pred) / wrong_pred)))
            self.all_amount_of_says.append(amount_of_say)

            # remake sampel weights
            for i in range(len(y)):
                if (y_hat[i] != y[i]):
                    sample_weights[i] = sample_weights[i] * math.exp(amount_of_say)
                else:
                    sample_weights[i] = sample_weights[i] * math.exp(-amount_of_say)
            
            normalize_val = sum(sample_weights)
            # normalize sample weights
            sample_weights = [w/normalize_val for w in sample_weights]
            # create new data based on new sample weights
            X,y = self.new_data(X,y,sample_weights)
    
    def new_data(self,X,y,sample_weights):
        """Function to generate new data as per sample weights"""
        total_samples = len(X)

        new_X = []
        new_y = []

        while(len(new_X) < total_samples):
            val = random.uniform(0,sum(sample_weights))

            if (val >= 0 and val < sample_weights[0]):
                new_X.append(X.iloc[0])
                new_y.append(y[0])
            else:
                for j in range(1,total_samples):
                    if (sum(sample_weights[:j-1]) <= val and val < sum(sample_weights[:j])):
                        new_X.append(X.iloc[j])
                        new_y.append(y[j])
                        break
        # print("pringint new_X:")
        # print(new_X)
        # print("len of new_x", len(new_X))
        # X_t = pd.concat(new_X)
        # X_t.reset_index(drop=True,inplace = True)
        # print("pringint X_T:")
        # print(X_t)
        new_X = pd.DataFrame(new_X)
        new_X.reset_index(drop=True,inplace = True)
        # print(new_X)
        new_y = pd.Series(new_y)

        return new_X,new_y

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = []
        
        pred_values = dict()

        for i in self.out_classes:
            pred_values[i] = 0

        for i in range(len(X)):
            tot_pred = 0
            for j in range(self.n_estimators):
                # predict using each estimator
                curr_y_hat = self.estimators_list[j].predict(X)

                # add the amount of say predicted same class of output vec
                pred_values[curr_y_hat[i]] += self.all_amount_of_says[j]
                # pred_values[self.estimators_list[j].predict(curr_sample)] += self.all_amount_of_says[j]
            
            # get the class with max amount of say 
            curr_max = 0
            curr_pred = self.out_classes[0]
            for key,value in pred_values.items():
                if (curr_max < value):
                    curr_max = value
                    curr_pred = key
            
            y_hat.append(curr_pred)
        
        y_hat = pd.Series(y_hat)
        return y_hat

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        plot_colors = "gb"
        plot_step = 0.02
        n_classes = 2
        for _ in range (self.n_estimators):
            plt.subplot(1, 3, _+1 )
            x_min, x_max = self.all_Xs[_].iloc[:, 0].min() - 1, self.all_Xs[_].iloc[:, 0].max() + 1
            y_min, y_max = self.all_Xs[_].iloc[:, 1].min() - 1, self.all_Xs[_].iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = self.estimators_list[_].predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(self.all_ys[_] == i)
                for i in range (len(idx[0])):
                    plt.scatter(self.all_Xs[_].loc[idx[0][i]][0], self.all_Xs[_].loc[idx[0][i]][1],c=color,cmap=plt.cm.PiYG, edgecolor='black', s=15)
        plt.suptitle("Decision Tree plots for individual estimators")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")

        plt.show()
        fig1 = plt

        # Figure 2
        plot_colors = "gb"
        plot_step = 0.02
        n_classes = 2
        x_min, x_max = self.data.iloc[:, 0].min() - 1, self.data.iloc[:, 0].max() + 1
        y_min, y_max = self.data.iloc[:, 1].min() - 1, self.data.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
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
        fig2 = plt

        return [fig1,fig2]
