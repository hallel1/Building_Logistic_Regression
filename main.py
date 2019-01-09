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
df=df_org.__deepcopy__()

print("Please be patient its loading... ")
if __name__ == "__main__":
    df2 = df.replace(np.nan, '', regex=True)# replace nan values with ''
    ###################################################################
    csv_org.insert_col_df(df2)
    csv_org.normalizationAll(df2)
    XMatrix = csv_org.x_matrix(df2)
    y = csv_org.y_vector(df2)
    X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size=0.25, random_state=42)
    v_theta = np.zeros(df.shape[1])  # logreg.random_theta(df2)
    betterTheta, L_thetaVec, countIter = logreg.lgReg(v_theta, X_train, y_train, alpha=0.001, maxIter=500,
                                                      difference=0.000000001)


    ##################################################################


    threshold=0.5
    TP, FN, FP, TN=logreg.predicted_Value(X_test, betterTheta, y_test,threshold)
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
    tpr=logreg.TPR(TP,FN)
    print("true positive rate (TPR)=",tpr)
    fpr=logreg.FPR(FP, TN)
    print("false positive rate(FPR)=",fpr)
    # ---------------------------------------------------------#



    # print graph roc_curve
    logreg.graph_L_theta(L_thetaVec, range(countIter))
    #Show error results on test set.
    logreg.roc_curve_graph(X_test, betterTheta,y_test)















