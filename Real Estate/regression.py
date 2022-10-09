import numpy
class Linear_Regression:
    #import numpy as np
    def __init__(self,alpha=0.01,epochs = 2000,lambdas=0,regularized=False):
        self.alpha = alpha
        self.epochs =epochs
        self.w = None
        self.b = None
        self.regularized = regularized
        self.lambdas=lambdas
        self.coef_ = self.w
        self.intercept_ = self.b
    def model(self,X):
        f_x = numpy.dot(X,self.w)+self.b
        return f_x
    def predict(self,X):
        train_size = len(X)
        y_pred = self.model(X)
        #cost = (y-y_pred)**2
        return y_pred
    def gradient(self,X,y,w,b):
        feature_size = len(X[0])
        train_size = len(X)
        dj_dw = numpy.zeros(feature_size)
        dj_db = 0
        y_pred = self.model(X)
        for i in range(feature_size):
            first_cost = y_pred-y
            cost = (first_cost)*X[:,i]
            if self.regularized:
                dj_dw[i] = (numpy.sum(cost)+(self.lambdas*self.w[i]))/train_size
            else:
                dj_dw[i] = numpy.sum(cost)/train_size
        dj_db = numpy.sum(first_cost)/train_size
        return dj_dw,dj_db
    def gradient_descent(self,X,y):
        feature_size = len(X[0])
        self.w = numpy.zeros(feature_size)
        self.b = 0
        for epoch in range(self.epochs):
            dj_dw,dj_db = self.gradient(X,y,self.w,self.b)
            self.w = self.w-(self.alpha*dj_dw)
            self.b = self.b-(self.alpha*dj_db)
        return self.w,self.b
    def fit(self,X,y):
            return self.gradient_descent(X,y)