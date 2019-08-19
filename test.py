import numpy as np

from SupervisedModels import LinearRegression



a = np.arange(4).reshape((2,2))
b = np.arange(4).reshape((2,2))
print(a*b)
print(a)
print(np.sum(a*b,axis=0))