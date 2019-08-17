import pandas as pd
import numpy as np
from Models import LinearRegression as linear


dataset = pd.read_csv("source/iris_numeric_lables.data")
df_X = dataset.iloc[:,0:4]
df_y = dataset.iloc[:,4]
model = linear.LinearRegression()

model.fit(df_X,df_y)

weight, weight_c = model.getWeights()
print (weight , weight_c)

print(model.predict(np.array([5.6,2.9,3.6,1.3])))
