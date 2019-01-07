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

    # a_list=[1,2,3]
    # b_list=[1,2,3]
    # z=[a * b for a, b in zip(a_list, b_list)]
    # print(z)
    # z=sum(z)
    # print (z)

    z=[a*b for a, b in zip(thata, X)]

    z = sum(z)
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
def lgReg_iter(thata, X,y):# X is vector of row
    h_of_xi=  h_func(thata,X)
    print(1- h_of_xi)
    calc= y * np.math.log(h_of_xi)+ (1-y)* np.math.log(1- h_of_xi)#######check!!
    return calc

#---------------------------
# def probability_func(thata,X_matrix,y):
#     h =h_func( thata,X)
#     prob=(h**y)*(1-h)**(1-y)
#     return prob

#---------------------------
def lgReg(thata, file):
    sum = 0

    count_row = file.shape[0]# parameter m
    for i in range(1, count_row + 1):
        yi = contain_row(i, file)
        yi = yi[:-1]
        X=contain_row(i,file)
        X=X[:-1]# all row except the last cell
        sum+=lgReg_iter(thata, X, yi)


    return 0

#---------------------------
def random_thata(df2):
    for i in range(df2.shape[1]):
        vector_thata=[]
        vector_thata.append(np.random.random())
        # print(i,vector_thata)
        return vector_thata


# #-----------MAIM---------------------
# if __name__ == "__main__":
#     print(1/(1+e**-2))
