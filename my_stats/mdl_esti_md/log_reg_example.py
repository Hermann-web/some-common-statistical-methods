'''
Author: Susan Li 
source: https://github.com/aihubprojects/Logistic-Regression-From-Scratch-Python/blob/master/LogisticRegressionImplementation.ipynb
'''

import numpy as np

import sys
import os.path
sys.path.append(os.path.abspath("."))
from my_stats.mdl_esti_md.prediction_metrics import PredictionMetrics

class LogisticRegression:
    
    # defining parameters such as learning rate, number ot iterations, whether to include intercept, 
    # and verbose which says whether to print anything or not like, loss etc.
    def __init__(self, learning_rate=0.01, num_iterations=50000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    # function to define the Incercept value.
    def __b_intercept(self, X):
        # initially we set it as all 1's
        intercept = np.ones((X.shape[0], 1))
        # then we concatinate them to the value of X, we don't add we just append them at the end.
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid_function(self, z):
        # this is our actual sigmoid function which predicts our yp
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, yp, y):
        # this is the loss function which we use to minimize the error of our model
        #return (-y * np.log(yp) - (1 - y) * np.log(1 - yp)).mean()
        return PredictionMetrics(y_true=y, y_pred_proba=yp, binary=True).log_loss(min_tol=True)
    
    # this is the function which trains our model.
    def fit(self, X, y):
        
        # as said if we want our intercept term to be added we use fit_intercept=True
        if self.fit_intercept:
            X = self.__b_intercept(X)
        
        # weights initialization of our Normal Vector, initially we set it to 0, then we learn it eventually
        self.W = np.zeros(X.shape[1])
        
        # this for loop runs for the number of iterations provided
        for i in range(self.num_iterations):
            
            # this is our W * Xi
            z = np.dot(X, self.W)
            
            # this is where we predict the values of Y based on W and Xi
            yp = self.__sigmoid_function(z)
            
            # this is where the gradient is calculated form the error generated by our model
            gradient = np.dot(X.T, (yp - y)) / y.size
            
            # this is where we update our values of W, so that we can use the new values for the next iteration
            self.W -= self.learning_rate * gradient
            
            # this is our new W * Xi
            z = np.dot(X, self.W)
            yp = self.__sigmoid_function(z)
            
            # this is where the loss is calculated
            loss = self.__loss(yp, y)
            
            # as mentioned above if we want to print somehting we use verbose, so if verbose=True then our loss get printed
            if(self.verbose ==True and i % 10000 == 0):
                print(f'step {round(100*i/self.num_iterations)}% loss: {loss} \t')
    
    # this is where we predict the probability values based on out generated W values out of all those iterations.
    def predict_prob(self, X):
        # as said if we want our intercept term to be added we use fit_intercept=True
        if self.fit_intercept:
            X = self.__b_intercept(X)
        
        # this is the final prediction that is generated based on the values learned.
        return self.__sigmoid_function(np.dot(X, self.W))
    
    # this is where we predict the actual values 0 or 1 using round. anything less than 0.5 = 0 or more than 0.5 is 1
    def predict(self, X):
        return self.predict_prob(X).round()

#data
import numpy as np
fit_intercept = False
loc, scale, size = 20, 3, 2000
sample = np.random.normal(loc=loc, scale=scale, size=size)
y = 12 + 2*sample #+ 3*power(sample, 2) + 0.0001*random.normal(0, scale, size)
y = (y>y.mean()).astype('int')
X = sample.reshape(-1, 1)

#model
model = LogisticRegression(learning_rate=0.1, num_iterations=500000, verbose=True)
model.fit(X, y)
print("pred_sc = ",(model.predict(X) == y).sum()/len(y))

preds = model.predict_prob(X)
cl = PredictionMetrics(y_true=y, y_pred_proba=preds, binary=True)
dd = {}
dd["acc"] = cl.get_binary_accuracy()
dd["rec"] = cl.get_recall_score()
dd["prec"] = cl.get_precision_score()
dd["conf"] = cl.get_confusion_matrix()
dd["log-likelihood"] = cl.compute_log_likelihood()
dd["log-loss"] = cl.log_loss()
print(dd)