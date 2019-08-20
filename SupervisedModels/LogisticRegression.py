import numpy as np

from SupervisedModels.LinearRegression import LinearRegression


class LogisticRegression(LinearRegression):

    def fit(self, records, label,a=0.0001,lamb=1,epsilon=0.00001):
        X = np.array(records)
        y = np.array(label)
        self.k = np.array(np.unique(y))
        # w(K,N)
        self.weight = np.ones((np.unique(y).shape[0],X.shape[1]))
        # w_c(K,1)
        self.weight_c = np.ones(np.unique(y).shape[0])
        # self.weight = np.ones(X.shape[1])
        # self.weight_c = 1

        for i in range(0,1000):
            # calculates cost sum[ h(N,1) - (N,1) ]=1
            #  c(1) = c(1) - mean[ h(M,1) - y(M,1) ]=1
            y_k = self.getWeight(y)
            weight_c_new = self.weight_c[y_k] - a*np.mean((self.h(X) - y))
            # w(N,1) = w(N,1) - divide[ (M,N)T.(M,1), M ]=(N,1)
            weight_new = self.weight[y_k] - a*np.mean(self.cost(X,y) + self.regularization(self.weight[y_k],lamb),axis=0)
            # update weights
            self.weight[y_k] = weight_new
            self.weight_c[y_k] = weight_c_new

    #y(M,1) = (?,1)
    def getWeight(self, y):
        return np.array(np.argwhere(y == self.k))

    # (M,N)T.(M,1)=(N,1)
    def cost(self,X, y):
        return X.T.dot(self.h(X) - y)

    def softmax(self, wX):
        # max Y [ X(M,K)/{sum Y [ X(M,K) ]=(M,1)} ]=(M,1)
        sum_prob = np.array([np.sum(np.exp(wX).T, axis=0)]).T
        #print(np.unique(np.divide(np.array(np.exp(wX)), sum_prob), axis=1))
        return np.array(np.argmax(np.divide(np.array(np.exp(wX)), sum_prob), axis=1))

    # M,1
    def h(self, X):
        # M , 1
        weight_key = self.softmax(self.w(X))
        if weight_key.size > 1:
            weight_key = max(weight_key)

        # sum y [ (M,N)*y(M,N)=(M,N) ]=(M,1) + (M,1)= (M,1)
        return np.sum(self.weight[weight_key]*X,axis=1) + self.weight_c[weight_key]
        # return X.dot(self.weight[weight_key].T) + self.weight_c[weight_key]

    # K = number of classes
    # X(M,N).(K,N).T=(M,K)
    def w(self,X):
        return X.dot(self.weight.T) + self.weight_c