# -*- coding: UTF-8 -*-
# @Time    : 15/11/2022
# @Author  : Marco Cheung
# @File    : Data2_Prepare_NoNormal.py
# @Software: BigDataPrediction - COMP
import pandas as pd
# import numpy as np


# Check if an unusual element appears in a column of data
def checkElement(dataset):
    for col in range(dataset.shape[1]):
        # training.csv - Row Index 0, 3, 4, 6, 9, 10, 14 are described in English words
        dataset_inCol = dataset[dataset.columns[col]].tolist()
        print(set(dataset_inCol))
        print("Column No=%d, Type Count: %d \n" % (col, len(set(dataset_inCol))))


# Find Error for Column Index 3 ('attribute 4')
def findError_Col3(dataset):
    errors = list()
    for row in range(len(dataset)):
        if (dataset.iloc[row, 3] != 'Sometimes') and (dataset.iloc[row, 3] != 'Always') and \
                (dataset.iloc[row, 3] != 'Frequently') and (dataset.iloc[row, 3] != 'no'):
            print("attribute 4 Error Located: (" + str(row) + ", 3), Error=" + str(dataset.iloc[row, 3]))
            errors.append(row)
    return errors


# Convert vocabulary describing frequency into numerical values
# e.g. For data2_training.csv, the vocabularyColumn is [0, 3, 4, 6, 9, 10, 14]
def convertVocabulary(dataset, vocabularyColumn):
    # Rules: 'yes' = 1; 'no' = 0
    # ['no', 'Sometimes', 'Frequently', 'Always'] = [0, 1, 2, 3]
    # ['A', 'B', 'M', 'P', 'W'] = [0, 1, 2, 3, 4]
    for col in vocabularyColumn:
        for row in range(len(dataset)):
            if dataset.iloc[row, col] == 'yes':
                dataset.iloc[row, col] = 1
            elif dataset.iloc[row, col] == 'no':
                dataset.iloc[row, col] = 0
            elif dataset.iloc[row, col] == 'Sometimes':
                dataset.iloc[row, col] = 1
            elif dataset.iloc[row, col] == 'Frequently':
                dataset.iloc[row, col] = 2
            elif dataset.iloc[row, col] == 'Always':
                dataset.iloc[row, col] = 3
            elif dataset.iloc[row, col] == 'A':
                dataset.iloc[row, col] = 0
            elif dataset.iloc[row, col] == 'B':
                dataset.iloc[row, col] = 1
            elif dataset.iloc[row, col] == 'M':
                dataset.iloc[row, col] = 2
            elif dataset.iloc[row, col] == 'P':
                dataset.iloc[row, col] = 3
            elif dataset.iloc[row, col] == 'W':
                dataset.iloc[row, col] = 4
    return dataset


'''
# Normalize columns with numbers, scale them to range(0, 1)
def normalization(dataset):
    for col in range(dataset.shape[1] - 1):
        # Get Max & Min value of this column
        max_in_col = dataset[str(dataset.columns[col])].max()
        min_in_col = dataset[str(dataset.columns[col])].min()

        print("Max:" + str(max_in_col))
        print("Min:" + str(min_in_col) + "\n")

        for row in range(0, len(dataset[str(dataset.columns[col])])):
            dataset.iloc[row, col] = (dataset.iloc[row, col] - min_in_col) / (max_in_col - min_in_col)

    return dataset
'''


def prepareTrainingData():
    sourceData = pd.read_csv("data2_training.csv")
    # sourceData_Arr = np.array(sourceData)
    # dataset_inRow = sourceData_Arr.tolist()

    checkElement(sourceData)  # Check if there are some unusual elements in each column
    errors = findError_Col3(sourceData)  # Find Error elements in Column 3 (index)

    # Drop rows with error element in Column 3 (index)
    for i in range(len(errors)):
        sourceData = sourceData.drop(index=errors[i])
    print(sourceData)

    # Converts the vocabulary describing frequency into numerical values to calculate the distance
    sourceData = convertVocabulary(sourceData, [0, 3, 4, 6, 9, 10, 14])

    # Normalization all the values (except labels) to range(0, 1)
    # sourceData = normalization(sourceData)

    sourceData.to_csv('data2_training_noNormal.csv', index=False, header=True)


def prepareValidationData():
    sourceData = pd.read_csv("data2_validation.csv")
    checkElement(sourceData)  # Validation Dataset has no error here

    # Converts the vocabulary describing frequency into numerical values
    sourceData = convertVocabulary(sourceData, [0, 3, 4, 6, 9, 10, 14])

    # Normalization all the values (except labels) to range(0, 1)
    # sourceData = normalization(sourceData)

    sourceData.to_csv('data2_validation_noNormal.csv', index=False, header=True)


def prepareTestData():
    sourceData = pd.read_csv("data2_test.csv")
    checkElement(sourceData)  # Test Dataset has no error here

    # Converts the vocabulary describing frequency into numerical values
    sourceData = convertVocabulary(sourceData, [0, 3, 4, 6, 9, 10, 14])

    # Normalization all the values (except labels) to range(0, 1)
    # sourceData = normalization(sourceData)

    sourceData.to_csv('data2_test_noNormal.csv', index=False, header=True)


if __name__ == "__main__":
    prepareTrainingData()
    # prepareValidationData()
    # prepareTestData()
