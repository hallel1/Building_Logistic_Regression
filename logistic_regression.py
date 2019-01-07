from math import e
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from csv_handle import contain_row
#from csv_handle import contain_col
import matplotlib.pyplot as plt






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
    for i in range(0, count_row ):
        yi = contain_row(i, file)
        yi = yi[-1]
        X=contain_row(i,file)
        X=X[:-1]# all row except the last cell
        sum+=lgReg_iter(thata, X, yi)


    return (X,yi,sum)

#---------------------------
def random_thata(df2):
    for i in range(df2.shape[1]):#range about num of col
        vector_thata=[]
        vector_thata.append(np.random.random())
        # print(i,vector_thata)
        return vector_thata


# #-----------MAIM---------------------
# if __name__ == "__main__":
#     print(1/(1+e**-2))
def plot(X,y,df):
    #count_row = df.shape[0]  # parameter m
    #for i in range(0, count_row)    :
    for i in range(df.shape[0]):  # range about num of row
       XRow =contain_row(i, df)
       X = X[:-1]  # all row except the last cell
    healthy = YCol[y == 1]
    # filter out the applicants that din't get admission
    not_admitted = df.loc[y == 0]

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')


    plt.plot(X,y, label='Decision Boundary')
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    plt.show()