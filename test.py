import numpy as np
import pandas as pd


from SupervisedModels import LinearRegression

ds = pd.read_csv("source/iris_n.data",header=None)

feature_a = ds.iloc[:,0:2]

mu = np.mean(feature_a,axis=0)
sigma = np.std(feature_a,axis=0)

normalised_a = (feature_a - mu) / sigma


#print(normalised_a.describe())

a = np.arange(8).reshape((4,2))

a = np.insert(a,0,1,axis=1)
print (a)