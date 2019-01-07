from math import e
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from csv_handle import contain_row







#--------------------Methods-----------------------------

#calc sigmond func  for calcing the  h  later
def sigmoid (z):
    g=1/(1+e**-z)
    return g

#----------------------------

def h_func( thata,X):
    z=thata*X
    g=sigmoid(z)
    return g

#----------------------------

# fun y^ - for Classification
def  classification( thata,X):
    h= h_func( thata,X)
    if h>=0.5:
        return 1
    else:
        return 0



"""
X- vector 
"""
#---------------------------
def lgReg_iter(thata, X,y, i):# X is vector of row
    h_of_xi=  h_func(thata,X)
    calc= y * np.math.log(h_of_xi)
    return 0

#---------------------------
# def probability_func(thata,X_matrix,y):
#     h =h_func( thata,X)
#     prob=(h**y)*(1-h)**(1-y)
#     return prob

#---------------------------
def lgReg(thata, X_matrix, Y,file):
    sum = 0
    count_row = file.shape[0]# parameter m
    for i in range(1, count_row + 1):
        X=contain_row(i)
        X=X[:-1]
        sum+=lgReg_iter(thata, X, Y, i)


    return 0



# #-----------MAIM---------------------
# if __name__ == "__main__":
#     print(1/(1+e**-2))
