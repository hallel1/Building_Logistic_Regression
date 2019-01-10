from math import e
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame
from csv_handle import contain_row
#from csv_handle import contain_col
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#--------------------Methods-----------------------------
# method that returnvector of  random_theta as the  col of num
def random_theta(df):
    vector_theta = []
    for i in range(df.shape[1]):#range about num of col
        vector_theta.append(np.random.random())
    return vector_theta


#-------------------------------

#calc sigmond func  for calcing the  h  later
def sigmoid (z):
    g=1/(1+e**-z)
    return g

#----------------------------
# method that  calc value(Qx) for the sigmond func
def h_func( theta,X):


    z=[a*b for a, b in zip(theta, X)]
    z = sum(z)
    g=sigmoid(z)
    return g


#------------------------------------------
# A method which predicts according to the calculation of whether the patient is sick or healthy
def  classification( theta,X,threshold):

    h= h_func( theta,X)
    if h>=threshold:
        return 1
    else:
        return 0

#---------------------------

# method that calc the cost for one  example
def lgReg_iter(theta, X,y):# X is vector of row
    h_of_xi=  h_func(theta,X)
    calc= y * np.math.log(h_of_xi)+ (1-y)* np.math.log(1- h_of_xi)#######check!!
    return calc

#---------------------------

# method that calc that cost , the sum for  all example
def cost(theta, x_train,y_train):#L(theta) func
    sum = 0

    count_row = len(y_train)#file.shape[0]# parameter m
    for i in range(0, count_row ):
        yi= y_train[i]
        X = x_train[i]#expecting get vector
        sum+=lgReg_iter(theta, X, yi)

    sum = (1/count_row) * sum
    return (sum)

#-------------------------------
#  method that calc the gradient - for improving theta
def gradient(theta,xi_vec,yi,numTrain):
    gradientVal=0
    h=h_func(theta, xi_vec)
    for j in range(len(xi_vec)):
        gradientVal+=(yi-h)*xi_vec[j]
    gradientVal=(1 / numTrain) *gradientVal
    return gradientVal
#------------------------------------------
def gradientDescentIter(theta,alpha,x_train,y_train):
    numTrain=len(y_train)
    for i in range(numTrain):
        gradientVal=gradient(theta,x_train[i],y_train[i],numTrain)
        theta=theta+alpha*gradientVal
    return theta
#------------------------------------------
# the main method that calc the most optimal theta  for getting optimal cost
def lgReg(theta, x_train,y_train,alpha, maxIter, difference):
    costVec=[]
    costVal = cost(theta, x_train, y_train)
    costVec.append(costVal)
    countIter=1 #count how match nodes there are in costVec list

    for i in range(maxIter):
        countIter=countIter+1
        theta=gradientDescentIter(theta, alpha,x_train,y_train)
        newCost = cost(theta, x_train, y_train)
        costVec.append(newCost)
        if abs(costVec[i]-newCost) <difference:#####checkkk
            print('yes diff', len(costVec))
            break

    return (theta,costVec,countIter)
#----------------------------------------------------
# method that  display L_theta graph
def graph_L_theta(costVec,vecIter):

    #plt.scatter(vecIter,costVec,label='skitscat',color='blue',marker='o',s=50)
    plt.plot(vecIter,costVec)
    plt.xlabel('iteration')
    plt.ylabel('L(theta)')
    plt.show()
#-----------------------------------------
# X_test- matrix of values
# h- predicated value
# y- real value (Vector of values)
# method that checks hwo many   true positive, true negative, false positive and false negative cases we have
#  when we given  a threshold the common threshold is 0.5
def predicted_Value (X_test,thata,Y_test,threshold):
    TP = 0  # was predicted that the patient is sick and was right ( y=1, h=1 )
    FN = 0  # the model predicted that the patient is healthy  and was wrong( y=1, h=0 )
    FP = 0  # the model predicted that the patient is sick and was wrong (y=0,h=1)
    TN = 0  # the model predicted that the patient is healthy and was right (y=0,h=0)
    for i in range(len(X_test)):
        xi=X_test[i]
        yi=Y_test[i]
        h=classification(thata, xi,threshold)
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
    if TP+FP ==0:# Prevents zero division
        return np.nan
    pre=TP/(TP+FP)
    return pre

#----------------------------------
def F_score(recall,precision):
    if recall == 0 or precision==0:# Prevents zero division
        return np.nan
    rec =1/recall
    pre=1/precision
    if rec+pre == 0:# Prevents zero division
        return np.nan
    f_score=2/(rec+pre)
    return f_score

#--------------------------------
# true positive rate-recall
def TPR(TP,FN):
    tpr=TP/(TP+FN)
    return tpr

#--------------------------------
## false positive rate
def FPR(FP,TN):
    fpr=FP/(FP+TN)
    return fpr

# --------------------------------
# num of threshols as num of  the example
def roc_curve_graph(X_test, thata, Y_test):
    num_threshold=20

    X=[]
    Y=[]
    threshold=0.01
    for i in range(num_threshold+1):
        threshold=threshold+0.04
        print("TP, FN, FP, TN")
        TP, FN, FP, TN=predicted_Value(X_test, thata, Y_test, threshold)

        print(TP, FN, FP, TN)
        X.append(FPR(FP,TN))
        Y.append( TPR(TP,FN))
    # plt.scatter(X, Y, label='skitscat', color='blue', marker='o', s=50)

    plt.plot(X,Y)
    print(X)
    print(Y)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title("ROC CURVE")
    plt.plot(X,Y,label='"ROC CURVE')
    plt.xlabel('x - FPR')
    plt.ylabel('y-TPR')

    plt.legend()
    plt.show()
    area=np.trapz(Y,X)
    area=abs(area)
    print(area)
    area=0
    for i in range(1,num_threshold+1):
        a=X[i-1]-X[i]
        b=a*Y[i-1]
        area=area+b
    print('area',area)






