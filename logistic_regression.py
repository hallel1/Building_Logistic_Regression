from math import e
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from csv_handle import contain_row
#from csv_handle import contain_col
import matplotlib.pyplot as plt


#--------------------Variables----------------------------
true=1
false=0
positive=1
negative=0



#--------------------Methods-----------------------------

#calc sigmond func  for calcing the  h  later
def sigmoid (z):
    g=1/(1+e**-z)
    return g

#----------------------------

def h_func( theta,X):

    # a_list=[1,2,3]
    # b_list=[1,2,3]
    # z=[a * b for a, b in zip(a_list, b_list)]
    # print(z)
    # z=sum(z)
    # print (z)

    z=[a*b for a, b in zip(theta, X)]
    z = sum(z)
    #z=np.dot(X, theta)
    #print('z')
    #print(z)
    g=sigmoid(z)
    return g

#----------------------------

# fun y^ - for Classification
def  classification( theta,X):
    h= h_func( theta,X)
    if h>=0.5:
        return 1
    else:
        return 0

#---------------------------
def lgReg_iter(theta, X,y):# X is vector of row
    h_of_xi=  h_func(theta,X)
    calc= y * np.math.log(h_of_xi)+ (1-y)* np.math.log(1- h_of_xi)#######check!!
    return calc

#---------------------------
# def probability_func(theta,X_matrix,y):
#     h =h_func( theta,X)
#     prob=(h**y)*(1-h)**(1-y)
#     return prob

#---------------------------
def lgReg(theta, file):
    sum = 0

    count_row = file.shape[0]# parameter m
    for i in range(0, count_row ):
        yi = contain_row(i, file)
        yi = yi[-1]
        X=contain_row(i,file)
        X=X[:-1]# all row except the last cell
        sum+=lgReg_iter(theta, X, yi)

    sum = -1 * sum #####check if ok to mult by -1
    sum = (1/count_row) * sum #####check if here!!!
    return (sum)

#---------------------------
def random_theta(df):
    vector_theta = []
    for i in range(df.shape[1]):#range about num of col
        vector_theta.append(np.random.random())
    return vector_theta


#-------------------------------
def gradient(theta, file,indexRow):
    gradientVal=0
   # count_col = file.shape[1]
    X = contain_row(indexRow, file)
    X = X[:-1]  # all row except the last cell
    yi = contain_row(indexRow, file)
    yi = yi[-1]
    m = file.shape[0]  # num of row
    h=h_func(theta, X)
    #gradientVal=(1 / m) * np.dot(X.transpose(), h - yi)####maybe need x.t (transpose)
    print('col',len(X))
    for j in range(len(X)):
        gradientVal+=(h - yi)*X[j]
    gradientVal=(1 / m) *gradientVal
    return gradientVal
#------------------------------------------
def gradientDescentIter(theta,alpha, file,indexRow):
    gradientVal=gradient(theta, file, indexRow)
    theta=theta+alpha*gradientVal
    return theta
#------------------------------------------
def gradientDescent(theta,alpha, file,indexRow,maxIter,difference):

    for j in range(maxIter):
        diffOrg= theta[0]
        theta=gradientDescentIter(theta, alpha, file, indexRow)
        print('new theta',theta)
        if abs(diffOrg-theta[0]) <difference:#####checkkk
            print('yes diff')
            break
    return theta
'''
def plot(X,y,df):
    #count_row = df.shape[0]  # parameter m
    #for i in range(0, count_row)    :
    listHealpy=[]
    for i in range(df.shape[0]):  # range about num of row
        row =contain_row(i, df)
        healthy = row[y == 1]
        #healthyX=healthy[:-1]# all row except the last cell
        print('healpy')
        print(healthy)

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
    '''

 # example how to draw graph
def print_graph():
    x=[2,5,6,7,8,9]
    y=[1,2,3,4,5,6]
    # plt.hist(x,y,histtype='bar',rwidth=0.8)#malben
    plt.scatter(x,y,label='skitscat',color='blue',marker='o',s=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

#------------------------------------------
def accuracy(right,all_test):
    a=right/all_test
    return a

#-----------------------------------
def error(accuracy):
    err=1-accuracy
    return err


#-----------------------------------