import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import csv_handle as csv_org
import logistic_regression as logreg
from sklearn.model_selection import train_test_split

##--------------------MAIN------------------------
#print('mainnnnnnnnn')
from csv_handle import contain_row

path = 'hearts.csv'
df_org = pd.read_csv(path)
# df_org.to_csv('changeData.csv')
# new_path = 'changeData.csv'
# df = pd.read_csv(new_path)
df=df_org.__deepcopy__()

#print(df_org)
# df.drop(df.columns[0], axis=1, inplace=True)  # because it add a new colum when we copy
# df.to_csv(new_path, encoding='utf-8', index=False)#
# print(df.shape)


if __name__ == "__main__":
    print(" check csv_org methods  ")



    # row= csv_org.contain_row(0,df)
    # print(row)
    #
    # row = csv_org.contain_row(0,df)
    # print(row)
    #
    col = csv_org.contain_col(3,df)

    # print(col)
    # col_ChestPain = csv_org.change_str_col('ChestPain',df)
    # print(col_ChestPain)
    # col = csv_org.contain_col(14,df)
    # print(col)
    # col_AHD = csv_org.change_str_col('AHD',df)
    # print(col_AHD)
    # col = csv_org.contain_col(13,df)
    # print(col)
    # col_Thal = csv_org.change_str_col('Thal',df)
    # print(col_Thal)
    # print( df['ChestPain'][1][1])
 #   df.to_csv(new_path, encoding='utf-8', index=False)
    # norCol = csv_org.normalization(df,2)
    # print(norCol)
  #
    df2 = df.replace(np.nan, '', regex=True)
  #  print(df2)
   # print(" *********")
    #csv_org.replaceNaN(df2, 12, '$')
   # csv_org.normalization(df2,12)
    #print(" ############!!!!!!")

#    csv_org.normalization(df2,3)
  #  csv_org.normalizationAll(df2)
   # print(" -------- ")
   # print(csv_org.contain_row(87,df2))
    ###################################################################3
    # df2.to_csv('changeData.csv')
    # new_path = 'changeData.csv'
    # df2.to_csv(new_path, encoding='utf-8', index=False)
    df2.insert(1, 'v', 1)
    df2.insert(5, 'typical',  np.nan)
    df2.insert(6, 'asymptomatic', np.nan)
    df2.insert(7, 'nonanginal', np.nan)
    df2.insert(8, 'nontypical', np.nan)
    #--------------------------------------------
    df2.insert(19, 'fixed', np.nan)
    # df2.insert(19, 'NaN', np.nan)
    df2.insert(20, 'normal', np.nan)
    df2.insert(21, 'reversable', np.nan)
    # --------------------------------------------
    # df2.insert(23, 'Yes', np.nan)
    # df2.insert(24, 'No', np.nan)
    csv_org.split_col_data('ChestPain', df2)
    csv_org.split_col_data('Thal', df2)
    csv_org.split_col_data('AHD', df2)
    # afther spliting the cols del the orginal
    df2.__delitem__('ChestPain')
    df2.__delitem__('Thal')
    # print(df2.shape)
    print(df2)




    # df2.to_csv(new_path, encoding='utf-8', index=False)

    ###################################################################
    col = csv_org.contain_col(3,df_org)



    

    # df2.to_csv(new_path, encoding='utf-8', index=False)
   # print(df2)
    csv_org.normalizationAll(df2)
    # df2.to_csv(new_path, encoding='utf-8', index=False)
  #  print(csv_org.recognizeColByNum(df2,15))
   # csv_org.normalization(df2,15)
   #  print(df2)

    # df2.to_csv(new_path, encoding='utf-8', index=False)
    # print(df2.shape)
    print(" check logistic regression  methods  ")



#    sum=logreg.lgReg(v_theta,df2)
#     print('sum',sum)
#     v_theta = logreg.random_theta(df2)
#     print('theta',v_theta)
#    betterTheta = logreg.gradientDescent(v_theta,0.9999999999999, df2,1,5,0.0000000000000003)

    #logreg.gradientDescentIter(v_theta,0.5, df2,1)
#    print('better', betterTheta)


    # sum=logreg.lgReg(v_theta,df2)
    # print('sum',sum)
    # print('theta',v_theta)
    #logreg.gradientDescentIter(v_theta,0.5, df2,1)
    # print('better', betterTheta)

    XMatrix=csv_org.x_matrix(df2)
    y=csv_org.y_vector(df2)
    # print('XMatrix.shape ' ,len(XMatrix),len(XMatrix[0]))
    # print('y.shape ', len(y))
    X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size = 0.25, random_state = 42)

    v_theta =np.zeros(df.shape[1])#logreg.random_theta(df2)
    #betterTheta = logreg.gradientDescent(v_theta, 0.9999999999999, df2, 1, 5, 0.0000000000000003)

    betterTheta, L_thetaVec,countIter= logreg.lgReg(v_theta, X_train,y_train,alpha=0.001, maxIter=500, difference=0.000000001)
    print('l len ',L_thetaVec)
    print('l len ',len(L_thetaVec))
    print('countIter ', countIter)
    logreg.graph_L_theta(L_thetaVec, range(countIter))




    #print('xt ',X_train)
    #print('xt[0] ', X_train[0])
    # h=logreg.h_func(betterTheta, X_test)
    # print(h)

    # logreg.classification(betterTheta, X_test)
#    logreg.plot(XRow, yi,df2)
  #  logreg.print_graph()

    # for i in range(len(X_test)):
    #     xi=X_train[i]
    #     # print(xi)
    #     h=logreg.classification(betterTheta, xi)
    #     print(i,h)
    #

    #########################################333
    #display errors



    threshold=0.5
    TP, FN, FP, TN=logreg.predicted_Value(X_test, betterTheta, y_test,threshold)
    print(TP, FN, FP, TN)

    right=TP+TN
    all_test=len(X_test)
    accuracy=logreg.accuracy(right, all_test)
    print("accuracy=",accuracy)
    error=logreg.error(accuracy)
    print("error=",error)
    recall=logreg.recall(TP,FN)
    print("recall=",recall)
    precision=logreg.precision(TP,FP)
    print("precision=",precision)
    f_score=logreg.F_score(recall,precision)
    print("F_score=",f_score)
    # true positive rate
    tpr=logreg.TPR(TP,FN)
    print("true positive rate (TPR)=",tpr)
    # false positive rate
    fpr=logreg.FPR(FP, TN)
    print("false positive rate(FPR)=",fpr)



    # print graph roc_curve
    # print (all_test)
    logreg.roc_curve_graph(X_test, betterTheta,y_test)














