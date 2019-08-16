import numpy as np

a = np.array([[2,1],[1, 2],[1,3]])
print(np.cov(a[0],a[1]))


x = a[0] - np.mean(a[0])
y = a[1] - np.mean(a[1])

def cov(x,y):
    covariance = np.empty([len(x),len(x)])
    for x_val in x:
        row = np.empty(len(x))
        for y_val in y:
            np.append(row, x_val * y_val / len(x))
            np.append(covariance,row)
    return covariance

print(cov(x,y))

