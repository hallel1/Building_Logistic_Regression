
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame


#------------------
typical=[1,0,0,0]
asymptomatic=[0,1,0,0]
nonanginal= [0,0,1,0]
nontypical=[0,0,0,1]
#----
yes=1
no=0
#----
fixed=[1,0,0,0]
NA= [0,1,0,0]
normal=[0,0,1,0]
reversable=[0,0,0,1]

#--------------------Methods-----------------------------

def contain_row(row_num,df):
    rowList = []
    for j in range(df.shape[1]):# run on the num of cols
        if j==0:
            continue
        rowList.append(df.iloc[row_num][j])

    return rowList
#---------------------------------------
def contain_col(col_num,df):
    colList = []
    for j in range(df.shape[0]):# run on the num of rows
        #col+=str(df.iloc[j][col_num])+" \n"
        colList.append(df.iloc[j][col_num])
    return colList

#---------------------------------------

def isNaN(val):
    return val != val or val==''
#-------------------------------
def averageCol(col):
    #didnt count nun lines, flagNaN is 1 if there are nan

    flagNaN=0;
    sum = 0
    line_count = 0
    for i in col:
        if not isNaN(i):
            sum += i
            line_count += 1
        else:
            flagNaN=1;

    if line_count == 0:
        return
    average = sum / line_count
    return (average,line_count,flagNaN)
#-------------------------------
def recognizeColByNum(file,col_num):
    #if col_num==0
#    print('j')
    header=list(file.columns.values)
    colName=header[col_num]
#    print(colName)
    return colName
#-------------------------------
def replaceNaN(file,col_num,avg):
    col = contain_col(col_num, file)  # create list from row
    rowNum=0
#    print('before')
#    print(col)
    colName= recognizeColByNum(file,col_num)

    for i in col:
        if isNaN(i):
            col = contain_col(col_num, file)
#            print(col)
 #           file.replace(np.nan, '$', regex=True)
#            print('alo')


            file[colName].at[rowNum] = avg
#            print(file[colName].at[rowNum])
            col = contain_col(col_num, file)
#            print(col)

            #file['Ca'].at[0] = "#"
           # file['Ca'].at[i] = '^^'
           # file.replace({'Ca': 0}, '#')
           #  file.replace('', '**', inplace=True)
           #  file.to_csv(path)
           #  file = pd.read_csv(path)
            #file.replace(0, '&')
            #file.set_value(i, col_num, avg)
        rowNum+=1
   # file.to_csv(path)
    #  file = pd.read_csv(path)

#    print('after')
#    print(col)
#-------------------------------
def normalization(file, col_num):
    col = contain_col(col_num, file)  # create list from row
    average,line_count,flagNaN=averageCol(col)
    if flagNaN==1:
#        print('there nannnnnnnnnnnn')
        replaceNaN(file,col_num, average)
        col = contain_col(col_num, file)  # create list from row
        average, line_count, flagNaN = averageCol(col)
    standard_deviation=0
    index=0
    normalization_col=col

#    print('nor fun')
#    print(col)

#    print('line_count '+str(line_count))


    for i in col:
        tmp =i-average
        standard_deviation+= tmp**2#pow
    standard_deviation/=line_count;
    standard_deviation = standard_deviation**(0.5)
#    print('standard_deviation '+str(standard_deviation))

    for i in col:
        tmp= i - average
        tmp/= standard_deviation
        normalization_col[index]=tmp
        index+=1
    colName = recognizeColByNum(file, col_num)
    file[colName]=normalization_col#Updating the column to be normalized
#    print(col)

  #  return normalization_col


# -------------------------------
def split_col_data(col_name,df2):
     length= df2.shape[0]
     col_num =df2.columns.get_loc(col_name)
     if col_name=='ChestPain':
         col=contain_col(col_num, df2)
         # print(col)
         for i in range(length):
             if  col[i]=='typical':
                 df2['typical'].at[i]=1
                 df2['asymptomatic'].at[i]=0
                 df2['nonanginal'].at[i]=0
                 df2['nontypical'].at[i]=0
             elif col[i]=='asymptomatic':
                 df2['typical'].at[i] = 0
                 df2['asymptomatic'].at[i] = 1
                 df2['nonanginal'].at[i] = 0
                 df2['nontypical'].at[i] = 0
             elif col[i] == 'nonanginal':
                 df2['typical'].at[i] = 0
                 df2['asymptomatic'].at[i] = 0
                 df2['nonanginal'].at[i] = 1
                 df2['nontypical'].at[i] = 0
             elif col[i] == 'nontypical':
                 df2['typical'].at[i] = 0
                 df2['asymptomatic'].at[i] = 0
                 df2['nonanginal'].at[i] = 0
                 df2['nontypical'].at[i] = 1
     if col_name =='Thal':
         col= contain_col(col_num,df2)
         for i in range(length):
             if col[i] == 'fixed':
                 df2['fixed'].at[i] = 1
                 df2['NaN'].at[i] = 0
                 df2['normal'].at[i] = 0
                 df2['reversable'].at[i] = 0
             elif isNaN(col[i]):
                 df2['fixed'].at[i] = 0
                 df2['NaN'].at[i] = 1
                 df2['normal'].at[i] = 0
                 df2['reversable'].at[i] = 0
             elif col[i] == 'normal':
                 df2['fixed'].at[i] = 0
                 df2['NaN'].at[i] = 0
                 df2['normal'].at[i] = 1
                 df2['reversable'].at[i] = 0
             elif col[i] == 'reversable':
                 df2['fixed'].at[i] = 0
                 df2['NaN'].at[i] = 0
                 df2['normal'].at[i] = 0
                 df2['reversable'].at[i] = 1
     if col_name =='AHD':
             col = contain_col(col_num, df2)
             for i in range(length):
                 if col[i] == 'Yes':
                     df2['AHD'].at[i] = 1
                     col[i] = yes
                 elif col[i] == 'No':
                     df2['AHD'].at[i] = 0
                     col[i] = no
# -------------------------------
'''
def normalizationAll(file):
    print('all')
    colNum=file.shape[1]
    rowNum = file.shape[0]
    for colIndex in range(colNum):  # run on the num of cols
        colName = recognizeColByNum(file, colIndex)
        colLine = contain_col(colIndex, file)
        print(colLine)
        print(type(colLine[0]))
        if colName=='ChestPain' or colName=='Thal' or colName=='AHD': #isinstance(colIndex[0], (list,)):#check if the col is list
            for j in range(rowNum):
              # print(colLine[j][])
               normalization(file, colLine[j])
        # else:
        #     normalization(file, col)
'''