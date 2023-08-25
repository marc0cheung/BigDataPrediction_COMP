# -*- coding: UTF-8 -*-
# @Time    : 15/11/2022
# @Author  : Marco Cheung
# @File    : DataCleaning.py
# @Software: BigDataPrediction_COMP
import pandas as pd
import numpy as np

# Read .csv file as Pandas DataFrame
sourceData = pd.read_csv("test_1.csv")

# loop every col except label col
# Use df.shape[1] to get count of col
for i in range(0, sourceData.shape[1] - 1):
    mean = sourceData[str(sourceData.columns[i])].mean()

    # loop every row of col(i)
    # if a element in (row[j],col[i]) is empty, fill it with col[i].mean()
    for j in range(0, len(sourceData[str(sourceData.columns[i])])):
        if pd.isnull(sourceData.iloc[j, i]):
            print(str(j) + ", " + str(i) + " is nan. Fill it with mean of its col.")
            sourceData.iloc[j, i] = mean

# Normalize dataset columns to range(0, 1)
# Find min & max value in each column
for col in range(0, sourceData.shape[1] - 1):
    # print(sourceData[str(sourceData.columns[col])])
    max_in_col = sourceData[str(sourceData.columns[col])].max()
    min_in_col = sourceData[str(sourceData.columns[col])].min()
    print("Max:" + str(max_in_col))
    print("Min:" + str(min_in_col) + "\n")

    for row in range(0, len(sourceData[str(sourceData.columns[col])])):
        sourceData.iloc[row, col] = (sourceData.iloc[row, col] - min_in_col) / (max_in_col - min_in_col)


print(sourceData)

sourceData.to_csv('data1_test_clean.csv', sep=',', index=False, header=True)
