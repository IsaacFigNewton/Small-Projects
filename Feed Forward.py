"""
Agenda:
************************************************************************************************************************
    Expand hidden layer to be any size
    Add functionality to add as many 1-node hidden layers as desired
    Expand functionality to have any number of varying size hidden layers (use a 2D array/matrix for layers? (make nodes and layers objects in C# version))
"""

#A simple neural network that learns to find the average of some numbers
import random
import numpy as np

#most important vars
#use inputSize of 2, outputSize of 1, and step size of 10 ** -10 for a perceptron
numLayers = 3

inputSize = 2
hiddenLayerSize = 2
outputSize = 2

stepSize = 10 ** -10
iterations = 2000

maxInValue = 10 ** 5

#instantiate the input list/input nodes
inputValues = [0] * inputSize

"""
instantiate the weights matrix, should be referenced by [weightLayerIndex][outputNodeIndex][inputNodeIndex]
the first number represents the number of layers

Ex: for a NN with 3 input nodes, 1 hidden layer with 2 nodes, and 4 output nodes, the arrays should look like this:
[
    #input layer to hidden layer
    [
        [w9, w10, w11]
        [w12, w13, w14]
    ]
    
    #hidden layer to output layer weights
    [
        [w1, w2]
        [w3, w4]
        [w5, w6]
        [w7, w8]
    ]
]

because there's only 1 hidden layer, there will only be 2 weight layers
"""
weights = [np.zeros((hiddenLayerSize, inputSize)), np.zeros((outputSize, hiddenLayerSize))]

#this loop should scale with the number of weight layers and thus hidden layers
#for each layer of weights
for i in range(len(weights)):
    #go through each sets of weights
    for j in range(len(weights[i])):
        #set each weight in each set
        for k in range(len(weights[i][j])):
            #set the corresponding weights (range from -1 to 1)
            weights[i][j][k] = random.randint(-1000, 1000)/1000.0

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

def sumWeightedValues(inputVals, inputWeights):
    #instantiate output variable
    output = 0.0

    #define it as the sum of weighted inputs
    for j in range(len(inputVals)):
        #RuntimeWarning: invalid value encountered in double_scalars
        output += inputVals[j] * inputWeights[j]

    return output

#cost function (square error) for 1 output node
def getCost(output, proutput):
    return (output - proutput) ** 2

#inputVals = list of the input values
#output = list of output values produced by NN
#proutput = list of target output values
def getCostSlopes(inputVals, hiddenVals, output, proutput):
    #make sure the arguments are right
    assert(len(inputVals) == inputSize)
    assert(len(output) == outputSize)
    assert(len(proutput) == outputSize)

    #define a matrix to store the slopes at the weights in sets corresponding to each output node,
    #   should be referenced by [weightLayerIndex][outputNodeIndex][inputNodeIndex]
    #the first number represents the number of layers
    #because there's only 1 hidden layer, there will only be 2 weight layers
    weightSlopes = [np.zeros((hiddenLayerSize, inputSize)), np.zeros((outputSize, hiddenLayerSize))]

    """
        In:   Hidden:   Out:
        O ------ O ------ O
         \      / \      /
          \    /   \    /
           \  /     \  /
            \/       \/
            /\       /\
           /  \     /  \
          /    \   /    \
         /      \ /      \
        O ------ O ------ O
        in2 wi22 hidden2 wh22 out2
        
    calculate the partial derivatives for all weights:
        notation:
            the letter after each w signifies the weight layer,
            the first number the resultant node number,
            the second number the input node number
            
        partial derivatives in the first weight layer:
            cost = (output1 - proutput1) ** 2
            output1 = hidden1 * wh11 + hidden2 * wh12 + ... + b
            output1 = (in1 * wi11 + in2 * wi12 + ...) * wh11 + (in1 * wi21 + in2 * wi22 + ...) * wh12 + ...
            d/dwi11(output1) = in1 * wh11
            and d/dwi23(output4) = in3 * wh42
            thus d/dwi11(cost1) = 2 * (output1 - proutput1) * (in1 * wh11)
            thus d/dwi1...(cost1) (partial derivative of the cost for output node 1 with respect to any weight connected
                to hidden node 1 in the 1st weights layer)
                = 2 * (output1 - proutput1) * (in... * wh1...)
            thus d/dwi...(cost1) (partial derivative of the cost for output node 1 with respect to any weight connected
                to any hidden node in the 1st weights layer)
                = 2 * (output1 - proutput1) * (in... * wh...)
            
        partial derivatives in the second weight layer:
            cost = (output - proutput) ** 2
            output1 = hidden1 * wh11 + hidden2 * wh12 + ... + b
            d/dwh11(output1) = hidden1
            thus d/dwh11(cost1) = 2 * (output1 - proutput1) * hidden1
            thus d/dwh11(cost1) = 2 * (output1 - proutput1) * (in1 * wi11 + in2 * wi12 + ...)
            and d/dwh1...(cost1) (partial derivative of the cost for output node 1 with respect to any weight connected
                to said output node in the 2nd weights layer)
                = 2 * (output1 - proutput1) * (in1 * wi...1 + in2 * wi...2 + ...)
            thus d/dwh...(cost...)  (partial derivative of the cost for any output node with respect to any weight
                connected to said output node in the 2nd weights layer) (hidden.. used for simplicity and references
                the hidden node linked to the desired output node by the weight in question)
                = 2 * (output... - proutput...) * hidden...
    """

    #there's bound to be an indexing bug somewhere in here

    #iterate through all the output nodes
    for o in range(outputSize):
        #iterate through all the weight layers
        for i in range(len(weightSlopes)):
            #iterate through all the sets in a layer (corresponding to resultant nodes)
            for j in range(len(weightSlopes[i])):
                #iterate through all the weights in each set
                for k in range(len(weightSlopes[i][j])):
                    #store each weight's corresponding slope based on the cost function

                    #if it's in the first weight layer, j = hidden node index, k = input node index
                    if (i == 0):
                        weightSlopes[0][j][k] = 2 * (output[o] - proutput[o]) * inputVals[k] * weights[1][o][j]
                    #if it's in the second weight layer, j = output node index, k = hidden node index
                    elif (i == 1):
                        weightSlopes[1][j][k] = 2 * (output[o] - proutput[o]) * hiddenVals[k]
                    else:
                        print("bruh momen")

    return weightSlopes

def backpropagate (weights, slopes, bias):
    #make sure the arguments are right

    assert(len(weights) == numLayers - 1)
    assert(len(slopes) == numLayers - 1)
    assert(len(bias) == outputSize)

    #modify every weight by its corresponding slope based on the cost function
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] -= stepSize * slopes[i][j][k]

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

    hiddenNodeVals = [0.0] * hiddenLayerSize
    #calculate values of hidden nodes
    for i in range(hiddenLayerSize):
        hiddenNodeVals[i] = sumWeightedValues(inputVals, weights[0][i])

    #calculate values of output nodes
    for i in range(outputSize):
        #calculate output
        output[i] = sumWeightedValues(hiddenNodeVals, weights[1][i]) + bias[i]

        #calculate predicted output (target output)
        proutput[i] = getAvg(inputVals)

    #get the slopes for all weights
    weightSlopes = getCostSlopes(inputVals, hiddenNodeVals, output, proutput)

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
"""
Tutorials and sources used to build this:
************************************************************************************************************************
    This guy's tutorials are how I learned to build the perceptron:     https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
    This helped with the weird, dumb stuff with arrays:                 https://stackoverflow.com/questions/72585703/modifying-2d-array-python
    
"""
