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


#---------------------------
def random_theta(df):
    vector_theta = []
    for i in range(df.shape[1]):#range about num of col
        vector_theta.append(np.random.random())
    return vector_theta


#-------------------------------
def gradient(theta,xi_vec,yi,numTrain):
    gradientVal=0
    h=h_func(theta, xi_vec)
    #gradientVal=(1 / m) * np.dot(X.transpose(), h - yi)####maybe need x.t (transpose)
    print('col',len(xi_vec))
    for j in range(len(xi_vec)):
        gradientVal+=(h - yi)*xi_vec[j]
    gradientVal=(1 / numTrain) *gradientVal
    return gradientVal
#------------------------------------------
def gradientDescentIter(theta,alpha,xi_vec,yi,numTrain):
    gradientVal=gradient(theta,xi_vec,yi,numTrain)
    theta=theta+alpha*gradientVal
    return theta
#------------------------------------------
def gradientDescent(theta,alpha,maxIter,difference,xi_vec,yi,numTrain):

    for j in range(maxIter):
        diffOrg= theta[0]
        theta=gradientDescentIter(theta, alpha,xi_vec,yi,numTrain)
        print('new theta',theta)
        if abs(diffOrg-theta[0]) <difference:#####checkkk
            print('yes diff')
            break
    return theta
#------------------------------------------
# fun y^ - for Classification
def  classification( theta,X):
    h= h_func( theta,X)
    if h>=0.5:
        return 1
    else:
        return 0

#---------------------------
# def probability_func(theta,X_matrix,y):
#     h =h_func( theta,X)
#     prob=(h**y)*(1-h)**(1-y)
#     return prob

#---------------------------
def x_matrix(file):

    X_mat=[]
    for i in range(file.shape[0]):  # run on the num of rows
        X_mat.append(xi_vector(file,i))
    return X_mat
def y_vector(file):
    yVec=[]
    for i in range(file.shape[0]):  # run on the num of cols
        yVec.append(yi_val(file,i))
    return yVec

def xi_vector(file,i):
    X = contain_row(i, file)
    X = X[:-1]  # all row except the last cell
    return X

def yi_val(file,i):
    yi = contain_row(i, file)
    yi = yi[-1]
    return yi
#---------------------------

def lgReg_iter(theta, X,y):# X is vector of row
    h_of_xi=  h_func(theta,X)
    calc= y * np.math.log(h_of_xi)+ (1-y)* np.math.log(1- h_of_xi)#######check!!
    return calc

#---------------------------

def lgReg(theta, x_train,y_train,alpha, maxIter, difference):
    costVal=cost(theta, x_train, y_train)
    index=0##########change
    betterTheta=gradientDescent(theta,alpha,maxIter,difference,x_train[index],y_train[index],len(y_train))
    costVal = cost(betterTheta, x_train, y_train)
    return costVal

def cost(theta, x_train,y_train):
    sum = 0

    count_row = len(y_train)#file.shape[0]# parameter m
    for i in range(0, count_row ):
        yi= y_train[i]
        X = x_train[i]#expecting get vector
        print('x_train[i] expecting get vector ',x_train[i])
        sum+=lgReg_iter(theta, X, yi)

    sum = -1 * sum #####check if ok to mult by -1
    sum = (1/count_row) * sum #####check if here!!!
    return (sum)
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
'''def print_graph():
    x=[2,5,6,7,8,9]
    y=[1,2,3,4,5,6]
    # plt.hist(x,y,histtype='bar',rwidth=0.8)#malben
    plt.scatter(x,y,label='skitscat',color='blue',marker='o',s=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()'''

#------------------------------------------
def accuracy(right,all_test):
    a=right/all_test
    return a

#-----------------------------------
def error(accuracy):
    err=1-accuracy
    return err

#----------------------------------
def recall(TP,FN):
    rec=TP/(TP+FN)
    return rec

#----------------------------------
def precision(TP,FP):
    pre=TP/(TP+FP)
    return pre

#----------------------------------
def F_score(recall,precision):
    rec =1/recall
    pre=1/precision
    f_score=2/(rec+pre)
    return f_score

#--------------------------------
# true positive rate
def TPR(TP,FN):
    tpr=TP/(TP+FN)
    return tpr

#--------------------------------
## false positive rate
def FPR(FP,TN):
    fpr=FP/(FP+TN)
    return fpr

# --------------------------------