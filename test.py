import numpy as np
import pandas as pd


a = np.zeros((3,6))

b = np.zeros((2,4))

c = [1,2,3]

stuff = [a.shape[1]] + c + [b.shape[1]]

print ( np.exp(stuff))


weights = []

for i in np.arange(len(stuff)-1):
    weights.append(np.random.rand(stuff[i], stuff[i + 1])*2 -1)

for i in np.arange(3,0,-1):
    print(i)

print (np.arange(4 - 2,0,-1))
