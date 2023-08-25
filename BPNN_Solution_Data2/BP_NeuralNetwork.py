# -*- coding: UTF-8 -*-
# @Time    : 15/11/2022
# @Author  : Marco Cheung
# @File    : BP_NeuralNetwork.py
# @Software: BigDataPrediction - COMP
import time
from math import exp
from math import tanh
from random import seed
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

errorList = []


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


# Initialize a neural network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()  # Create Empty List for BN_NN
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Neuron Activation Calculation
def activate(weights, inputs):
    # Assumes the bias is the last weight in the list of weights
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation = activation + (weights[i] * inputs[i])

    return activation


# Transfer neuron activation using Sigmoid/ReLU Function
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))  # Sigmomid
    # return max(0.0, activation)  # ReLU Activation
    # return tanh(activation)


# From Network input to Network output - Forward Propagate
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)  # Create new list for output sigmoid results
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Use sigmoid to transfer derivative
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Updating Weights
def weights_update(network, row, learningRate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learningRate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learningRate * neuron['delta']


# Network Training
def train_network(network, train, learningRate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error = sum_error + sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            # sum_error = sum_error + sum([np.power((expected[i] - outputs[i]), 2) for i in range(len(expected))])
            backward_propagate(network, expected)
            weights_update(network, row, learningRate)
        print('>epoch=%d, learningRate=%.3f, err=%.3f' % (epoch, learningRate, sum_error))
        errorList.append(sum_error)


# Use trained network to predict
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def trainNeuralNetwork_Selected(network, trainingSet, learningRate, epoches, n_outputs):
    # START NEURAL NETWORK TRAINING
    startTime = time.time()
    print("\nSTART TRAINING ......\n")
    train_network(network, trainingSet, learningRate, epoches, n_outputs)
    print("\n====================TRAINING END====================")
    print("TRAINING RESULT:")
    print(network)

    trainedWeights = np.array(network)  # Saved trained network weights to np.array
    np.save('weights.npy', trainedWeights)
    print(">TRAINED WEIGHTS SAVED TO: weights.npy")
    executionTime = (time.time() - startTime)  # End Timing - Training Ends
    print("Training Time: %f" % executionTime)

    plt.plot(errorList)  # Draw ErrorSum
    plt.show()

    # Predict use trained network (See the accuracy after training)
    print("\n\n====================START TESTING USING LABEL in TRAINING DATASET====================\n")
    right_count = 0
    for row in trainingSet:
        prediction = predict(network, row)
        if row[-1] == prediction:
            right_count = right_count + 1

        print('Expected=%d, Got=%d' % (row[-1], prediction), end='\r')
    print('Accuracy (training.csv)=%.3f percent' % (right_count * 100 / len(trainingSet)))
    return network


def validateNetwork_Selected(weightsFilename, validationSet):
    startTime = time.time()

    # Load network from .npy file
    readin = np.load(weightsFilename, allow_pickle=True)
    network = readin.tolist()

    # Predict use validation dataset to see the F1-Scores after training
    print("\n\n====================START TESTING USING LABEL in VALIDATION DATASET====================\n")
    right_count = 0
    groundTruth = []
    predResults = []
    for row in validationSet:
        prediction = predict(network, row)
        groundTruth.append(row[-1])
        predResults.append(prediction)

        if row[-1] == prediction:
            right_count = right_count + 1

        print('Expected=%d, Got=%d' % (row[-1], prediction), end='\r')
    print('Accuracy (validation.csv)=%.3f percent' % (right_count * 100 / len(validationSet)))
    print("\ngroundTruth=")
    print(groundTruth)
    print("predResults=")
    print(predResults)
    print("\nMicro-F1 Score: " + str(f1_score(groundTruth, predResults, average='micro')))
    print("Macro-F1 Score: " + str(f1_score(groundTruth, predResults, average='macro')))

    executionTime = (time.time() - startTime)  # End Timing - Validation Ends
    print("Evaluating Time: %f" % executionTime)


def getTestResult_Selected(weightsFilename, testSet):
    startTime = time.time()

    # Load network from .npy file
    readin = np.load(weightsFilename, allow_pickle=True)
    network = readin.tolist()

    # Predict result for test.csv using trained network
    print("\n\n====================Predict Test.csv with Training Dataset====================\n\n")
    predResults_test = []
    for row in testSet:
        prediction = predict(network, row)
        predResults_test.append(prediction)

    print(predResults_test)

    executionTime = (time.time() - startTime)  # End Timing - Validation Ends

    # Write the prediction results to test.csv and save to a new csv file
    test_dataFrame = pd.read_csv('data2_test.csv')
    for row in range(len(test_dataFrame)):
        test_dataFrame.iloc[row, 15] = predResults_test[row]  # Get the last col of each row

    test_dataFrame.to_csv('data2_test_result.csv', sep=',', index=False, header=True)
    print("Results saved to: data2_test_result.csv")

    print("Evaluating Time: %f" % executionTime)


def main():
    seed(1)

    # Read in cleaned .csv files
    dataset = csv_reader('data2_training_clean.csv')
    validation_dataset = csv_reader('data2_validation_clean.csv')
    test_dataset = csv_reader('data2_test_clean.csv')

    # Set BP Network argvs
    learningRates = 0.07
    epoch = 2500
    n_inputs = len(dataset[0]) - 1  # How many input of this dataset
    n_outputs = len(set([row[-1] for row in dataset]))  # How many types of output in total
    n_hidden = 25

    network = initialize_network(n_inputs, n_hidden, n_outputs)
    # readin = np.load("weights_2000_25_0p07.npy", allow_pickle=True)  # Continue training based on weights(epoch=2000)
    # network = readin.tolist()

    # PRINT INITIAL NETWORK INFORMATION
    print("INIT. NETWORK: n_Input=" + str(n_inputs) + ", n_Output=" + str(n_outputs) + ", n_Hidden=" + str(n_hidden))
    print("\nINITIAL WEIGHTS:")
    print(network)

    # Use console to navigate users' behaviors
    print('\n========== COMPXXXX - BIG DATA COMPUTING - Neural Network Demo ==========')
    print('===================== Marco Cheung XXXXXXXXX =====================\n')
    print('[1] Train Neural Network with data2_training_clean.csv')
    print('[2] Validate & Get F1-Scores with data2_validation_clean.csv')
    print('[3] Get Prediction Results from data2_test.csv')
    flag = int(input('\nFunction Select: '))

    if flag == 1:
        trainNeuralNetwork_Selected(network, dataset, learningRates, epoch, n_outputs)
    elif flag == 2:
        validateNetwork_Selected('weights.npy', validation_dataset)
    elif flag == 3:
        getTestResult_Selected('weights.npy', test_dataset)
    else:
        print("Wrong Input!")


if __name__ == "__main__":
    main()
