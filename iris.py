import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from SupervisedModels import LinearRegression as Linear_Model

from SupervisedModels import LogisticRegression as Log_model

# checks accuracy
def validate(y_pred, y_real):
    return np.divide(np.array(np.where(y_pred == y_real)).size, y_real.shape[0])


def linear(x,y):
    model = Linear_Model.LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    weight, weight_c = model.getWeights()
    print(weight, weight_c)
    print(validate(y_pred, y))


def logistic(x,y):
    model = Log_model.LogisticRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    weight, weight_c = model.getWeights()
    print(weight, weight_c)
    print(validate(y_pred, y))
    print(model.predict(np.array([5.1,3.4,1.5,0.23])))


# PCa stuff
def lieaner_pca(x,y):
    # PCa stuff
    model_pca = Linear_Model.LinearRegression()
    pca = PCA(.975)
    pca.fit(x)
    model_pca.fit(pca.transform(x), y)
    y_pred = model_pca.predict(pca.transform(x))
    weight, weight_c = model_pca.getWeights()
    print(weight, weight_c)
    print(validate(y_pred, y))
    print(pca.explained_variance_ratio_)




dataset = pd.read_csv("source/iris_n.data")
df_X = dataset.iloc[:,0:4]
df_y = dataset.iloc[:,4]

logistic(df_X,df_y)
