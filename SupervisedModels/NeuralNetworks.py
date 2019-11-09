import numpy as np

class NeuralNetworks:

    def fit(self, records, label, a=0.3, lamb=0.05,hiddenlayer=[4,4,4]):

        # x(M,N)
        X = np.array(records)
        # y(M,K)
        y = np.zeros((records.shape[0],np.unique(label).shape[0]))
        for k in np.unique(label):
            y[np.array(np.where(label == k)), k] = 1

        self.layers = [X.shape[1]] + hiddenlayer + [y.shape[1]]
        # theta(L,?,?)
        self.weights = []
        self.bias = []
        for i in np.arange(len(self.layers) - 1):
            self.weights.append(np.random.rand(self.layers[i], self.layers[i + 1])*2 - 1)
            self.bias.append((np.random.rand(1, self.layers[i + 1])*2 - 1))


        for k in range(0,100000):
            alpha = X
            # alphasets(L-1,?)
            alphasets = [alpha]
            for i in np.arange(len(self.layers) - 1):
                # alpha(M,?) = alpha(M,?).theta(?,?)
                alpha = self.g(alpha.dot(self.weights[i]) + self.bias[i])
                alphasets.append(alpha)

            delta = y - alpha
            # deltasets(L-1,?)
            deltasets = [delta]
            for i in np.arange(len(self.layers) - 2,0,-1):
                # delta(M,?) = delta(M,?).theta(?,?).T*alpha(?,?)*(1-alpha(?,?))
                # g'(z) = alpha(?,?)*(1-alpha(?,?))
                delta = delta.dot(self.weights[i].T)*alphasets[i]*(1-alphasets[i])
                deltasets.append(delta)
            deltasets.reverse()

            for i in np.arange(len(self.layers) - 1):
                # theta +=  a*(alpha[l].T.deltasets[l+1])/M - lambda*theta/M
                self.weights[i] += np.divide(a*(alphasets[i].T.dot(deltasets[i])) - lamb*self.weights[i], X.shape[0])
                self.bias[i] += a*np.mean(deltasets[i],axis=0)

            if k % 500 == 0:
                print("cost:", self.cost(X,y,lamb))

    def cost(self, X, y, lamb):
        sumWieghts = 0
        for i in self.weights:
            sumWieghts += np.sum(i)
        return np.mean(-np.log(self.h(X)) * y - np.log(1 - self.h(X)) * (1 - y)) + sumWieghts*lamb/(2*X.shape[0])

    # h(X) = [M,1]
    def h(self, X):
        alpha = X
        for i in np.arange(len(self.layers) - 1):
            alpha = self.g(alpha.dot(self.weights[i]) + self.bias[i])
        return np.array(alpha)

    def g(self, z):
        return np.power(1 + np.exp(-z), -1)

    def predict(self, X):
        # (M,1)
        return np.argmax(self.h(X).T, axis=0)

    def getWeights(self):
        return self.weights, self.bias