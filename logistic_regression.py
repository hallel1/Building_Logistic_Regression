from math import e
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from csv_handle import contain_row
#from csv_handle import contain_col
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#--------------------Methods-----------------------------

#calc sigmond func  for calcing the  h  later
def sigmoid (z):
    g=1/(1+e**-z)
    return g

#----------------------------

def h_func( theta,X):


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
        gradientVal+=(yi-h)*xi_vec[j]###check y-h or h-y
    gradientVal=(1 / numTrain) *gradientVal
    return gradientVal
#------------------------------------------
def gradientDescentIter(theta,alpha,xi_vec,yi,numTrain):
    gradientVal=gradient(theta,xi_vec,yi,numTrain)
    theta=theta+alpha*gradientVal
    return theta
#------------------------------------------
'''def gradientDescent(theta,alpha,maxIter,difference,xi_vec,yi,numTrain):

    for j in range(maxIter):
     #   diffOrg= theta[0]
        theta=gradientDescentIter(theta, alpha,xi_vec,yi,numTrain)
    #    print('new theta',theta)
    #    if abs(diffOrg-theta[0]) <difference:#####checkkk
    #        print('yes diff')
            break
    return theta'''
#------------------------------------------
# fun y^ - for Classification
def  classification( theta,X,threshold):

    h= h_func( theta,X)
    if h>=threshold:
        return 1
    else:
        return 0

#---------------------------
# def probability_func(theta,X_matrix,y):
#     h =h_func( theta,X)
#     prob=(h**y)*(1-h)**(1-y)
#     return prob

#---------------------------


def lgReg_iter(theta, X,y):# X is vector of row
    h_of_xi=  h_func(theta,X)
    calc= y * np.math.log(h_of_xi)+ (1-y)* np.math.log(1- h_of_xi)#######check!!
    return calc

#---------------------------

def lgReg(theta, x_train,y_train,alpha, maxIter, difference):
    costVec=[]
    costVal = cost(theta, x_train, y_train)
    costVec.append(costVal)
    index=0##########change

    for i in range(maxIter):

        theta=gradientDescentIter(theta, alpha,x_train[index],y_train[index])
        newCost = cost(theta, x_train, y_train)
        costVec.append(newCost)
        if abs(costVec[i]-newCost) <difference:#####checkkk
    #        print('yes diff')
            break

    # betterTheta=gradientDescent(theta,alpha,maxIter,difference,x_train[index],y_train[index],len(y_train))

    return (theta,costVec)
#----------------------------------------------------

def cost(theta, x_train,y_train):#L(theta) func
    sum = 0

    count_row = len(y_train)#file.shape[0]# parameter m
    for i in range(0, count_row ):
        yi= y_train[i]
        X = x_train[i]#expecting get vector
        print('x_train[i] expecting get vector ',x_train[i])
        sum+=lgReg_iter(theta, X, yi)

    #####check if ok to mult by -1
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
    # print('col',len(X))
    for j in range(len(X)):
        gradientVal+=(yi-h)*X[j]
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
        # print('new theta',theta)
        if abs(diffOrg-theta[0]) <difference:#####checkkk
            print('yes diff')
            break
    return theta


#

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


def graph_L_theta(costVec,vecIter):

    plt.scatter(costVec,vecIter,label='skitscat',color='blue',marker='o',s=50)
    plt.xlabel('iteration')
    plt.ylabel('L(theta)')
    plt.legend()
    plt.show()
#-----------------------------------------
# X_test- matrix of values
# h- predicded value
# y- real value (Vector of values)
#
def predicted_Value (X_test,thata,Y_test,threshold):
    TP = 0  # was predicted that the patient is sick and was right ( y=1, h=1 )
    FN = 0  # the model predicted that the patient is healthy  and was wrong( y=1, h=0 )
    FP = 0  # the model predicted that the patient is sick and was wrong (y=0,h=1)
    TN = 0  # the model predicted that the patient is healthy and was right (y=0,h=0)
    for i in range(len(X_test)):
        xi=X_test[i]
        yi=Y_test[i]
        h=classification(thata, xi,threshold)
        #print(i,h)
        if yi == 1 and h == 1:
            TP = TP+1
        elif yi == 1 and h == 0:
            FN = FN+1
        elif yi == 0 and h == 1:
            FP =FP +1
        elif yi == 0 and h == 0:
            TN =TN +1
    return (TP,FN,FP,TN)


#--------------------------------------
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
# true positive rate-recall
def TPR(TP,FN):
    tpr=TP/(TP+FN)
    return tpr

#--------------------------------
## false positive rate- precision
def FPR(FP,TN):
    fpr=FP/(FP+TN)
    return fpr

# --------------------------------
# num of threshols as num of  the example
def roc_curve_graph(X_test, thata, Y_test):
    num_threshold=20


    X=[]
    Y=[]
    threshold=0
    for i in range(num_threshold+1):
        threshold=threshold+0.1
        print(threshold)

        TP, FN, FP, TN=predicted_Value(X_test, thata, Y_test, threshold)
        print(TP, FN, FP, TN)
        X.append(FPR(FP,TN))
        Y.append( TPR(TP,FN))
    print(X)
    print(Y)
    plt.scatter(X, Y, label='skitscat', color='blue', marker='o', s=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()




