"""
Agenda:
************************************************************************************************************************
    Expand output layer to be any size (use a 2D array/matrix for weights)
        Note: the weights for all output nodes seem to be backpropagating by
            the slopes of the weights connected to the last output node
    Create a 1-node hidden layer
    Expand hidden layer to be any size (use a 2D array/matrix for weights)
    Add functionality to add as many 1-node hidden layers as desired
    Expand functionality to have any number of varying size hidden layers (use a 2D array/matrix for layers? (make nodes and layers objects in C# version))
"""

#A simple neural network that learns to find the average of some numbers
import random

#most important vars
#use inputSize of 2, outputSize of 1, and step size of 10 ** -10 for a perceptron
inputSize = 2
outputSize = 2
stepSize = 10 ** -10
iterations = 1000

maxInValue = 10 ** 5

#instantiate the input list/input nodes
inputValues = [0] * inputSize

#instantiate the input weights matrix, should be referenced by [outputNodeIndex, inputNodeIndex]
weights = [[0.0] * inputSize] * outputSize

#define the corresponding weights (range from -1 to 1)
for i in range(outputSize):
    #instantiate 1 set of weights corresponding to the input nodes (1 weight/node) for each output node
    inputNodeWeights = [0.0] * inputSize
    #define each weight in the set of weights
    for j in range(inputSize):
        inputNodeWeights[j] = random.randint(-1000, 1000)/1000.0
    weights[i] = inputNodeWeights

#for debugging
OGWeights = str(weights)

bias = [0.0] * outputSize
proutput = [0.0] * outputSize
output = [0.0] * outputSize


#set guiding function to train 1 output node
def getAvg (inputVals):
    #add up all inputs
    sum = 0
    for val in inputVals:
        sum += val

    #get the average
    return float(sum) / inputSize

#get output for 1 output node
def getOutput(inputVals, inputWeights, bias):
    #instantiate output variable
    output = 0.0

    #define it as the sum of weighted inputs
    for j in range(inputSize):
        output += inputVals[j] * inputWeights[j]

    #add the bias
    output += bias

    return output

#cost function (square error) for 1 output node
def getCost(output, proutput):
    return (output - proutput) ** 2

#inputVals = list of the input values
#output = list of output values produced by NN
#proutput = list of target output values
def getCostSlopes(inputVals, output, proutput):
    #make sure the arguments are right
    assert(len(inputVals) == inputSize)
    assert(len(output) == outputSize)
    assert(len(proutput) == outputSize)

    #define a matrix to store the slopes at the weights in sets corresponding to each output node, should be referenced by [outputNodeIndex, inputNodeIndex]
    weightSlopes = [[0.0] * inputSize] * outputSize

    """
        In:   Out:
        O ---- O
          \   /
           \ /
            X
           / \
          /   \
        O ---- O
        
    calculate the partial derivatives for all weights with respect to each weight:
        cost = (output - proutput) ** 2
        output1 = in1 * w1 + in2 * w2 + ... + b
        d/dw1(output1) = in1
        thus d/dw1(cost1) = 2 * (output1 - proutput1) * in1
        and d/dw...(cost1) = 2 * (output1 - proutput1) * in...
        thus d/dw...(cost...) = 2 * (output... - proutput...) * in...
    """

    #iterate through all the output nodes' weight sets
    for i in range(outputSize):
        #iterate through all the weights in each set
        for j in range(inputSize):
            #...and store it in a list of weights' corresponding slopes based on the cost function
            weightSlopes[i][j] = 2 * (output[i] - proutput[i]) * inputVals[j]

    return weightSlopes

def backpropagate (weights, slopes, bias):
    #make sure the arguments are right
    assert(len(weights) == outputSize)
    assert(len(slopes) == outputSize)
    assert(len(bias) == outputSize)

    #modify every weight by its corresponding slope based on the cost function
    for i in range(outputSize):
        for j in range(inputSize):
            weights[i][j] -= stepSize * slopes[i][j]

    #modify the bias by its partial derivative
    for k in range(outputSize):
        bias[k] -= stepSize * 2 * (output[k] - proutput[k])

#assumes m1 and m2 are lists of the same length
def performMatrixOperation (m1, operation, m2):
    resultingMatrix = [0.0] * len(m1)

    #This is where functions in Java would come in handy
    if (operation == "+"):
        for i in range(len(m1)):
            resultingMatrix[i] = m1[i] + m2[i]
    elif (operation == "-"):
        for i in range(len(m1)):
            resultingMatrix[i] = m1[i] - m2[i]
    elif (operation == "*"):
        for i in range(len(m1)):
            resultingMatrix[i] = m1[i] * m2[i]
    elif (operation == "/"):
        for i in range(len(m1)):
            resultingMatrix[i] = m1[i] / m2[i]
    else:
        print("bruh, cringe")

    return resultingMatrix


#define the training procedure for 1 round of training
def train (iterationNum, inputVals):

    #calculate output stuff for every output node
    for i in range(outputSize):
        #calculate output
        output[i] = getOutput(inputVals, weights[i], bias[i])

        #calculate predicted output (target output)
        proutput[i] = getAvg(inputVals)

    #get the slopes for all weights
    weightSlopes = getCostSlopes(inputVals, output, proutput)

    #backpropagate
    backpropagate(weights, weightSlopes, bias)

    #Print stuff every few iterations
    if (iterationNum % 100 == 0):
        print("Running another round of training... (Iteration " + str(iterationNum) + ")\n"
              "******************************************************")

        # Print input values
        print("Input values: " + str(inputVals))
        print("Original weights: " + OGWeights)
        print("Weights: " + str(weights))
        print("Bias: " + str(bias))

        # print output/backpropagation info
        print("Expected output: " + str(getAvg(inputVals)))
        print("Actual output: " + str(output))
        print("Difference: " + str(performMatrixOperation(proutput, "-", output)))

        print ("Backpropagating...")
        print("Weight slopes: " + str(weightSlopes))

        #print an empty line to make the debugging look nice :)
        print("")

#run the NN
if __name__ == '__main__':
    print('Starting program...')

    print("Training...")
    for i in range(iterations + 1):

        #define the input values
        for j in range(0, inputSize):
            inputValues[j] = random.randint(0, maxInValue)

        train(i, inputValues)

#Tutorials and sources used to build this:
#   This guy's tutorials are how I learned to build the perceptron: https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
