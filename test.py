import numpy as np

from SupervisedModels import LinearRegression

a = np.ones(10).reshape(5,2)


print (a[np.array(np.argwhere(a == 1))])