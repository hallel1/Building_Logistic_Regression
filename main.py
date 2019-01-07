import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
import csv_handle as csv_org
import logistic_regression as logreg
##--------------------MAIN------------------------
#print('mainnnnnnnnn')
path = 'data.csv'
name = 'ChestPain'
df_org = pd.read_csv(path)
# df_org.to_csv('changeData.csv')
# new_path = 'changeData.csv'
# df = pd.read_csv(new_path)
df=df_org.__deepcopy__()

#print(df_org)
# df.drop(df.columns[0], axis=1, inplace=True)  # because it add a new colum when we copy
# df.to_csv(new_path, encoding='utf-8', index=False)#
print(df.shape)


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
    df2.to_csv('changeData.csv')
    new_path = 'changeData.csv'
    df2.to_csv(new_path, encoding='utf-8', index=False)
    df2.insert(4, 'typical',  np.nan)
    df2.insert(5, 'asymptomatic', np.nan)
    df2.insert(6, 'nonanginal', np.nan)
    df2.insert(7, 'nontypical', np.nan)
    #--------------------------------------------
    df2.insert(18, 'fixed', np.nan)
    df2.insert(19, 'NaN', np.nan)
    df2.insert(20, 'normal', np.nan)
    df2.insert(21, 'reversable', np.nan)
    # --------------------------------------------
    # df2.insert(23, 'Yes', np.nan)
    # df2.insert(24, 'No', np.nan)





    df2.to_csv(new_path, encoding='utf-8', index=False)

    ###################################################################
    col = csv_org.contain_col(3,df_org)
    print(col)
    csv_org.split_col_data('ChestPain', df2)
    csv_org.split_col_data('Thal',  df2)
    csv_org.split_col_data('AHD', df2)
    # afther spliting the cols del the orginal
    df2.__delitem__('ChestPain')
    df2.__delitem__('Thal')
    # df2.to_csv(new_path, encoding='utf-8', index=False)
    # print(df2.shape)
    print(" check logistic regression  methods  ")
    v_thata=logreg.random_thata(df2)
    logreg.lgReg(v_thata,df2)












    print(" finsh check csv_org methods  ")
    #print(df)