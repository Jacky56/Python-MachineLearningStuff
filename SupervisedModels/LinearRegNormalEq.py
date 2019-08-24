import numpy as np
from numpy.linalg import inv

# O(N^3) compute time
class LinearNormal:

    def __init__(self):
        self.weight = np.array([])
        self.loss_old = 0

    def fit(self, records, label):
        # X = np.insert(records, 0, 1, axis=1)
        X = np.concatenate((np.ones(records.shape[0]).reshape(records.shape[0],1),records),axis=1)
        y = label
        self.weight = inv(X.T.dot(X)).dot(X.T.dot(y))

    def predict(self, X):
        # X = np.insert(X, 0, 1, axis=1)
        X = np.concatenate((np.ones(X.shape[0]).reshape(X.shape[0],1),X),axis=1)
        return np.round(self.h(X))

    def h(self, X):
        return X.dot(self.weight)

    def getWeights(self):
        w_c = self.weight[0]
        w = self.weight[1:]
        return w, w_c
