# %%
import csv
from math import log
from collections import defaultdict, Counter
import random
import numpy as np

# set random seed so that random draws are the same each time
random.seed(12409)

# %%
# compute ridge estimates given X, y, and Lambda
def ridge(X, y, fLambda):
	
	# TODO: compute afBeta
	
	#return afBeta
    return

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

for i, row in enumerate(reader):
    y = float(row[0])
    x = [float(val) for val in row[1:]]


    rando = random.randint(0,1)
    if rando == 0:
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

# %%
beta = ridge(X_matrix_demeaned, Y_matrix_demeaned, 0) # (Modify the last parameter to change lambda)
# print beta


# TODO: compute accuracy of your estimates