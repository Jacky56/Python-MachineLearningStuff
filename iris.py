import pandas as pd
import numpy as np
from SupervisedModels import LinearRegression as linear

#checks accuracy
def validate(y_pred, y_real):
    return 1 - np.divide(np.sum(np.abs(y_pred - y_real)), y_real.shape[0])

dataset = pd.read_csv("source/iris_numeric_lables.data")
df_X = dataset.iloc[:,0:4]
df_y = dataset.iloc[:,4]
model = linear.LinearRegression()

model.fit(df_X,df_y)

weight, weight_c = model.getWeights()

y_pred = model.predict(df_X)

print(validate(y_pred, df_y))