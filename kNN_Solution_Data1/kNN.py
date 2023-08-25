# -*- coding: UTF-8 -*-
# @Time    : 15/11/2022
# @Author  : Marco Cheung
# @File    : kNN.py
# @Software: BigDataPrediction_COMP
import time
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def csv_reader(filename):
    # Read csv file & convert rows to list
    sourceData = pd.read_csv(filename)
    sourceData_Arr = np.array(sourceData)
    dataset = sourceData_Arr.tolist()
    # dataset[0][-1] = int(dataset[0][-1])

    # the last element of the sub-list should be an integer
    for i in range(len(dataset)):
        if str(dataset[i][-1]) != 'nan':
            dataset[i][-1] = int(dataset[i][-1])

    # print(dataset)
    return dataset


# Calculate Euclidean Distance
def euclideanDist(row1, row2):
    distance = 0.0
    # the last col in each row is a class label which is ignored
    for i in range(len(row1) - 1):
        distance = distance + (row1[i] - row2[i]) ** 2

    return sqrt(distance)


# Calculate Manhattan Distance
def manhattanDist(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance = distance + abs(row1[i] - row2[i])

    return distance


# Calculate Chebyshev Distance
def chebyshevDist(row1, row2):
    distance = []
    # the last col in each row is a class label which is ignored
    for i in range(len(row1) - 1):
        distance.append(abs(row1[i] - row2[i]))

    return max(distance)


# Find the most similar neighbors
def findNeighbors(train, test_row, num_neighbors):
    distance = list()
    for train_row in train:
        dist = euclideanDist(test_row, train_row)  # Use Euclidean Distance
        # dist = manhattanDist(test_row, train_row)  # Use Manhattan Distance
        # dist = chebyshevDist(test_row, train_row)  # Use Chebyshev Distance
        distance.append((train_row, dist))

    distance.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distance[i][0])
    return neighbors


# classification with neighbors
def classification(train, test_row, num_neighbors):
    neighbors = findNeighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    # A majority vote method
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def main():

    # Read training file as database of kNN
    k = 1
    train_dataset = csv_reader("training_1.csv")
    validation_dataset = csv_reader("validation_1.csv")
    test_dataset = csv_reader("test_1.csv")
    # test_dataset = csv_reader("data1_test_clean.csv")

    # prediction = classification(dataset, dataset[0], 3)
    # print('Expected %d, Got %d' % (dataset[0][-1], prediction))

    # Verify Results using labels in training dataset
    correctCount = 0
    for row in range(len(train_dataset)):
        prediction = classification(train_dataset, train_dataset[row], k)
        if train_dataset[row][-1] == prediction:
            correctCount = correctCount + 1

        print('Expected %d, Got %d' % (train_dataset[row][-1], prediction), end='\r')
    accuracy = correctCount / len(train_dataset) * 100
    print("[training.csv] Rows: " + str(row) + ", Accuracy: " + str(accuracy))

    # Verify Results using labels in validation.csv
    startTime = time.time()  # Start Timing - Validate Starts
    correctCount_Validation = 0
    groundTruth = []
    predResults = []
    for row in range(len(validation_dataset)):
        prediction = classification(train_dataset, validation_dataset[row], k)
        groundTruth.append(validation_dataset[row][-1])  # Add groundTruth and prediction to list
        predResults.append(prediction)

        if validation_dataset[row][-1] == prediction:
            correctCount_Validation = correctCount_Validation + 1

        print('Expected %d, Got %d' % (validation_dataset[row][-1], prediction), end='\r')

    accuracy = correctCount_Validation / len(validation_dataset) * 100
    print("[validation.csv] Rows: " + str(row) + ", Accuracy: " + str(accuracy))
    print("\ngroundTruth=")
    print(groundTruth)
    print("predResults=")
    print(predResults)
    print("\nMicro-F1 Score: " + str(f1_score(groundTruth, predResults, average='micro')))
    print("Macro-F1 Score: " + str(f1_score(groundTruth, predResults, average='macro')))

    executionTime = (time.time() - startTime)  # End Timing - Validation Ends
    print("Evaluating Time: %f" % executionTime)

    # Predict results in test.csv using training data
    print("\n\n====================Predict Test.csv with Training Dataset====================\n\n")
    startTime_test = time.time()  # Start Timing - Test starts
    predResults_test = []
    for row in range(len(test_dataset)):
        prediction = classification(train_dataset, test_dataset[row], k)
        predResults_test.append(prediction)

    print(predResults_test)
    executionTime_test = (time.time() - startTime)  # End Timing - Validation Ends
    print("Testing Time: %f" % executionTime_test)

    # Write the prediction results to test.csv and save to a new csv file
    test_dataFrame = pd.read_csv('test_1.csv')
    for row in range(len(test_dataFrame)):
        test_dataFrame.iloc[row, 17] = predResults_test[row]  # Get the last col of each row

    test_dataFrame.to_csv('data1_test_result.csv', sep=',', index=False, header=True)
    print("Results saved to: data1_test_result.csv")


if __name__ == "__main__":
    main()
