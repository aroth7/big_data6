# %%
import csv
from math import log
from collections import defaultdict, Counter
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set random seed so that random draws are the same each time
random.seed(12409)

# %%
# compute ridge estimates given X, y, and Lambda
def ridge(X, y, fLambda):
    # split into different parts so easier to debug
    p = X.transpose()
    b = np.matmul(p, X)
    
    iden = fLambda * np.identity(X.shape[1])
    inside_parens =  np.add(b , iden)
    inverse_of_inside = np.linalg.inv(inside_parens)
    first_dot_product = np.matmul(inverse_of_inside, p)
    afBeta = np.matmul(first_dot_product, y)
    return np.array(afBeta)
def predict(beta, X):
    return np.array(np.dot(X, beta.flatten()))
def acc(y_pred, y_true):
    sum = 0
    for i in range(len(y_pred)):
        sum+=(y_pred[i] - y_true[i])**2
    return sum/len(y_pred)

loans = []
# %%
# NOTE: changed this to processed_data.csv because loans_ridge doesn't include all our variables
f = open('processed_data.csv', 'r')

reader = csv.reader(f)
header = next(reader)

X = []
Y = []
testing_X = []
testing_Y = []

coeffs= [random.randint(0,100000) for _ in range(33)]
for i, row in enumerate(reader):
    y = row[1]
    x = [float(val) for val in row]
    # print(x)
    # break
    x.pop(0)
    x.pop(0)
    x = [a*b for a,b in zip(coeffs, x)]

    rando = random.random()
    if rando > 0.5:
        X.append(x)
        Y.append(y)
        
    else:
       testing_X.append(x)
       testing_Y.append(y)

f.close()

X_matrix  = np.matrix(X).astype(float)
Y_matrix  = np.array(Y).astype(float)
testing_X = np.matrix(testing_X).astype(float)
testing_Y = np.array(testing_Y).astype(float)

X_matrix_demeaned  = X_matrix  - np.mean(X_matrix, 0)
Y_matrix_demeaned  = Y_matrix  - np.mean(Y_matrix)
testing_X_demeaned = testing_X - np.mean(testing_X, 0)
testing_Y_demeaned = testing_Y - np.mean(testing_Y)
X_matrix_demeaned  = np.array(X_matrix_demeaned)
Y_matrix_demeaned  = np.array(Y_matrix_demeaned)
testing_X_demeaned = np.array(testing_X_demeaned)
testing_Y_demeaned = np.array(testing_Y_demeaned)

# %%
x = []
y = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []
train = []
test = []
for i in range(1, 5000, 50):
    beta = ridge(X_matrix_demeaned, Y_matrix_demeaned, i).flatten() # (Modify the last parameter to change lambda)
    # print beta
    x.append(i)
    y.append(beta[0])
    y2.append(beta[1])
    y3.append(beta[2])
    y4.append(beta[3])
    y5.append(beta[4])
    y6.append(beta[8])

    print(beta)
    y_pred = predict(beta, testing_X_demeaned).flatten()
    y_train_pred = predict(beta, X_matrix_demeaned).flatten()
    test_accu = acc(y_pred, testing_Y_demeaned)
    train_accu = acc(y_train_pred, Y_matrix_demeaned)
    train.append(train_accu)
    test.append(test_accu)
    print('test',test_accu)
    print('train',train_accu)
    
# TODO: compute accuracy of your estimates
# plt.plot(x,y)
# plt.plot(x,y2)
# plt.plot(x,y3)
# plt.plot(x,y4)
# plt.plot(x, y5)
# plt.plot(x, y6)
# plt.xlabel("Lambda")
# plt.ylabel("Coefficients")

plt.plot(x,train,label='train')
plt.plot(x,test,label='test')
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.legend(loc="upper left")
plt.show()