import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from SupervisedModels import LinearRegression as Linear_Model
from SupervisedModels import LinearRegNormalEq as Linear_norm_Model
from SupervisedModels import LogisticRegression as Log_model

# checks accuracy
def validate(y_pred, y_real):
    return np.divide(np.array(np.where(y_pred == y_real)).size, y_real.shape[0])

def mean_normalise(X):
    mu = np.mean(X,axis=0)
    std = np.std(X)
    return (X - mu) / std


def useModel(model,X,y):
    model.fit(X, y)
    y_pred = model.predict(X)
    print(validate(y_pred, y))
    return model.getWeights()

# PCa stuff
def pca(model,X,y,infoGain=0.975):
    # PCa stuff
    pca = PCA(infoGain)
    pca.fit(X)
    model.fit(pca.transform(X), y)
    y_pred = model.predict(pca.transform(X))
    print(validate(y_pred, y))
    print(pca.explained_variance_ratio_)
    return model.getWeights()




dataset = pd.read_csv("source/iris_n.data", header=None)
df_X = dataset.iloc[:,0:4]
df_y = dataset.iloc[:,4]

df_X = mean_normalise(df_X)

linear = Linear_Model.LinearRegression()
logistic = Log_model.LogisticRegression()
linear_norm = Linear_norm_Model.LinearNormal()


#print(useModel(linear_norm, df_X, df_y))
print(pca(linear, df_X, df_y,0.99))