import numpy as np


class LinearRegression:

    def __init__(self):
        self.weight = np.array([])
        self.weight_c = 0
        self.cost_old = 0

    def fit(self, records, label,a=0.001,lamb=2,epsilon=0.00001):
        X = np.array(records)
        y = np.array(label)
        self.weight = np.ones(X.shape[1])
        self.weight_c = 1

        for i in range(0,10000):
            # calculates cost sum[ h(N,1) - (N,1) ]=1
            cost = np.mean((self.h(X) - y)**2)
            if abs(cost - self.cost_old) < epsilon:
                print("cost contains minor changes:", i)
                break

            #  c(1) = c(1) - mean[ h(M,1) - y(M,1) ]=1
            weight_c_new = self.weight_c - a*np.mean((self.h(X) - y))
            # w(N,1) = w(N,1) - divide[ (M,N)T.(M,1), M ]=(N,1)
            weight_new = self.weight - a*np.divide(X.T.dot(self.h(X) - y) + self.regularization(self.weight,lamb), X.shape[0])
            # update weights
            self.weight = weight_new
            self.weight_c = weight_c_new

            self.cost_old = cost

    #
    def sigmoid(self, X):
        1 == 1

    # cannot be implemented
    def softmax(self, X):
        return np.max(np.exp(self.h(X)))

    def regularization(self,weight,lamb):
        return lamb*np.sum(weight**2)


    def predict(self, X):
        return np.round(self.h(X))

    def toNumeric(self,data):
        numericLabel = np.unique(data)
        data_new = data
        for i in range(0, len(numericLabel)):
            np.put(data_new, np.where(data == numericLabel[i]), [i])
        return data_new, numericLabel

    def loss(self, X_record, y_value):
        return self.h(X_record) - y_value

    # X(M,N).W(N,1)=h(M,1)
    def h(self, X):
        return X.dot(self.weight) + self.weight_c

    def getWeights(self):
        return self.weight, self.weight_c