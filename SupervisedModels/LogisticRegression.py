import numpy as np

from SupervisedModels.LinearRegression import LinearRegression


class LogisticRegression(LinearRegression):

    def fit(self, records, label,a=0.00001,lamb=1,epsilon=0.00001):

        # w(K,N)
        self.weight = np.ones((np.unique(label).shape[0],records.shape[1]))
        # w_c(K,1)
        self.weight_c = np.ones(np.unique(label).shape[0])
        # self.weight = np.ones(X.shape[1])
        # self.weight_c = 1

        for k in np.unique(label):
            k_index = list(np.argwhere(k == label).flatten())
            y = np.array(label[k_index])
            X = np.array(records.iloc[k_index])
            for i in range(0,10000):

                # calculates cost sum[ h(N,1) - (N,1) ]=1
                #  c(1) = c(1) - mean[ h(M,1) - y(M,1) ]=1
                argk = k-1
                weight_c_new = self.weight_c[argk] - a*np.mean((self.h(X) - k))
                # w(N,1) = w(N,1) - divide[ (M,N)T.(M,1), M ]=(N,1)
                weight_new = self.weight[argk] - a * np.divide(self.cost(X, k) + lamb*self.weight[argk], X.shape[0])
                # update weights
                self.weight[argk] = weight_new
                self.weight_c[argk] = weight_c_new


    # (M,N)T.(M,1)=(N,1)
    def cost(self,X, y):
        return X.T.dot(self.h(X) - y)

    def softmax(self, wX):
        # max Y [ X(M,K)/{sum Y [ X(M,K) ]=(M,1)} ]=(M,1)
        sum_prob = np.array([np.sum(np.exp(wX).T, axis=0)]).T
        return np.array(np.argmax(np.divide(np.array(np.exp(wX)), sum_prob), axis=1))

    # M,1
    def h(self, X):
        # M , 1
        weight_key = self.softmax(self.w(X))
        return np.sum(self.weight[weight_key]*X,axis=1) + self.weight_c[weight_key]
        #return X.dot(self.weight[weight_key].T) + self.weight_c[weight_key]

    # K = number of classes
    # X(M,N).(K,N).T=(M,K)
    def w(self,X):
        return X.dot(self.weight.T) + self.weight_c
