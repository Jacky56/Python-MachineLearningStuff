import numpy as np

from SupervisedModels.LinearRegression import LinearRegression


class LogisticRegression(LinearRegression):

    def fit(self, records, label,a=0.001,lamb=1):
        # x(M,N)
        X = np.array(records)
        # y(M,K)
        y = np.zeros((records.shape[0],np.unique(label).shape[0]))
        for k in np.unique(label):
            y[np.array(np.where(label == k)), k] = 1

        # w(K,N)
        self.weight = np.ones((np.unique(label).shape[0], records.shape[1]))
        # w_c(K,1)
        self.weight_c = np.ones(np.unique(label).shape[0])

        for i in range(0,10000):
            # calculates cost sum[ h(N,1) - (N,1) ]=1
            # w_c(K,1) = w_c(K,1) - sum x [ h(M,K) - y(M,K) ]=(1,K).T
            weight_c_new = self.weight_c - a*np.sum((self.h(X) - y)/X.shape[0], axis=0).T
            # w(N,1) = w(N,1) - divide[ (M,N).T.(M,K), M ]=(N,K).T
            weight_new = self.weight - a*np.divide(X.T.dot(self.h(X) - y).T + lamb*self.weight, X.shape[0])
            # update weights
            self.weight = weight_new
            self.weight_c = weight_c_new

            if i % 1000 == 0:
                print(self.cost(X,y))

    #1
    def cost(self,X,y):
        # sum [ h(M,K)*y(M,k) - h(M,K)*y(M,k) ]= 1
        return np.mean(-np.log(self.h(X))*y - np.log(1 - self.h(X))*(1 - y))

    # h(M,K)
    def softmax(self, wX):
        # X(M,K)/{sum Y [ X(M,K) ]=(M,1)} =(M,K)
        sum_prob = np.array([np.sum(np.exp(wX).T, axis=0)]).T
        return np.array(np.divide(np.array(np.exp(wX)), sum_prob))

    # M,K
    def h(self, X):
        # M , K
        return self.softmax(self.w(X))

    # (M,1)
    def predict(self, X):
        # (M,1)
        return np.argmax(self.h(X).T, axis=0)

    # K = number of classes
    # X(M,N).(K,N).T=(M,K)
    def w(self,X):
        return X.dot(self.weight.T) + self.weight_c
