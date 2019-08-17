import numpy as np


class LinearRegression:

    def __init__(self):
        self.weight = np.array([])
        self.weight_c = 0

    def fit(self, records, label,a=0.001,lamb=1,epsilon=0.0001):
        X = np.array(records)
        y = np.array(label)
        self.weight = np.ones(X.shape[1])
        self.weight_c = 1

        for i in range(0,1000):
            # calculates cost sum(h(N,1))=1
            cost = np.mean((self.h(X) - y)**2)

            #  c(1) = c(1) - mean[ h(M,1) - y(M,1) ]
            weight_c_new = self.weight_c - a*np.mean((self.h(X) - y))
            # w(N,1) = w(N,1) - divide[ (M,N)T.(M,1), M ]=(N,1)
            weight_new = self.weight - a*np.divide(X.T.dot(self.h(X) - y) + self.regularization(self.weight,lamb), X.shape[0])
            # update weights
            self.weight = weight_new
            self.weight_c = weight_c_new

    def sigmoid(self, h_value):
        1 == 1

    def regularization(self,weight,lamb):
        return lamb*np.sum(weight**2)

    def predict(self, X):
        return self.h(X)

    def loss(self, X_record, y_value):
        return self.h(X_record) - y_value

    # X(M,N).W(N,1)=h(M,1)
    def h(self, X):
        return X.dot(self.weight) + self.weight_c

    def getWeights(self):
        return self.weight, self.weight_c
