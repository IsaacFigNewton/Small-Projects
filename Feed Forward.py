"""
Agenda:
************************************************************************************************************************
    Add functionality to add as many 1-node hidden layers as desired
    Expand functionality to have any number of varying size hidden layers (use a 2D array/matrix for layers? (make nodes and layers objects in C# version))
"""

#A simple neural network that learns to find the average of some numbers
import random
import numpy as np

#most important vars
#use inputSize of 2, outputSize of 1, and step size of 10 ** -10 for a perceptron


inputSize = 4
numHiddenLayers = 1
hiddenLayerSize = 4
outputSize = 3

stepSize = 10 ** -10
iterations = 5000

maxInValue = 10 ** 3

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

because there's only 1 hidden layer, there will only be 2 weight sets
"""

if numHiddenLayers <= 0:
    weights = [np.zeros((outputSize, inputSize))]
elif numHiddenLayers == 1:
    weights = [np.zeros((hiddenLayerSize, inputSize)), np.zeros((outputSize, hiddenLayerSize))]
elif numHiddenLayers > 1:
    weights = [np.zeros((hiddenLayerSize, inputSize)),
                    np.zeros((hiddenLayerSize, hiddenLayerSize)) * (numHiddenLayers - 1),
                    np.zeros((outputSize, hiddenLayerSize))]

#this loop should scale with the number of weight sets and thus hidden layers
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

def sumWeightedValues(vals, weights):
    assert(len(vals) == len(weights))

    #instantiate output variable
    output = 0.0

    #define it as the sum of weighted inputs
    for i in range(len(vals)):
        #RuntimeWarning: invalid value encountered in double_scalars
        output += vals[i] * weights[i]

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
    assert(hiddenVals is None or len(hiddenVals) == numHiddenLayers)
    assert(len(output) == outputSize)
    assert(len(proutput) == outputSize)
    assert(len(weights) > 0)

    #define a multi-dimensional array to store the slopes at the weights in sets corresponding to each output node,
    #   should be referenced by [weightLayerIndex][resultantNodeIndex][inputNodeIndex]
    weightSlopes = []

    if hiddenVals == None or numHiddenLayers <= 0:
        weightSlopes.append(np.zeros((outputSize, inputSize)))

    elif numHiddenLayers == 1:
        weightSlopes.append(np.zeros((hiddenLayerSize, inputSize)))
        weightSlopes.append(np.zeros((outputSize, hiddenLayerSize)))

    elif numHiddenLayers > 1:
        weightSlopes.append(np.zeros((hiddenLayerSize, inputSize)))
        for i in range(numHiddenLayers):
            weightSlopes.append(np.zeros((hiddenLayerSize, hiddenLayerSize)))
        weightSlopes.append(np.zeros((outputSize, hiddenLayerSize)))

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
            the letter after each w signifies the weight set,
            the first number the resultant node number,
            the second number the input node number
            
        partial derivatives in the first weight set:
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
            
        partial derivatives in the second weight set:
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

    numWeightSets = len(weights)

    #iterate through all the output nodes
    for o in range(outputSize):
        frstPrtOfPrtlDrvtvs = 2 * (output[o] - proutput[o])

        #iterate through all the weight sets
        for i in range(len(weightSlopes)):
            #iterate through all the sets in a layer (corresponding to resultant nodes)
            for j in range(len(weightSlopes[i])):
                #iterate through all the weights in each set
                for k in range(len(weightSlopes[i][j])):

                    #store each weight's corresponding slope based on the cost function
                    # slope function is the partial derivative of the respective
                    if (numWeightSets == 1):
                        weightSlopes[0][o][k] += frstPrtOfPrtlDrvtvs * inputVals[k]

                    elif (numWeightSets == 2):
                        # if it's in the first weight set, j = hidden node index, k = input node index
                        if (i == 0):
                            #slope function is the partial derivative of the respective
                            weightSlopes[0][j][k] += frstPrtOfPrtlDrvtvs * inputVals[k] * weights[1][o][j]
                        #if it's the last weight set
                        else:
                            weightSlopes[1][j][k] += frstPrtOfPrtlDrvtvs * hiddenVals[0][k]

                    #if there're more than 2 weight sets
                    else:
                        #if it (the partial derivative desired) is in the 1st weight set
                        if (i == 0):
                            #slope function is the partial derivative of the cost function with respect to the weights
                            weightSlopes[0][j][k] += frstPrtOfPrtlDrvtvs\
                                                     * inputVals[k]\
                                                     * getSubsequentWeightsSum(weightSlopes,
                                                                               i, j, o)
                        #if it's the 2nd weight set, j = hidden node index, k = input node index
                        if (i == 1):
                            #slope function is the partial derivative of the respective
                            weightSlopes[1][j][k] += frstPrtOfPrtlDrvtvs \
                                                     * hiddenVals[0][j] \
                                                     * getSubsequentWeightsSum(weightSlopes,
                                                                               i, j, o)
                        #if it's in any subsequent weight set,
                        else:
                            weightSlopes[i][j][k] += frstPrtOfPrtlDrvtvs \
                                                     * getPriorNodesSum(hiddenVals, i, k) \
                                                     * getSubsequentWeightsSum(weightSlopes,
                                                                               i, j, o)
                            # weightSlopes[1][j][k] += frstPrtOfPrtlDrvtvs * hiddenVals[i - 2][k]
                        # weightSlopes[0][j][k] += frstPrtOfPrtlDrvtvs * inputVals[k] * weights[1][o][j]

    return weightSlopes

#gets the sum of the corresponding previous nodes and their weights
# for the partial derivatives of weights in NN's with more than 1 weight set
def getPriorNodesSum (hiddenVals, currentWeightLayer, currentInputNode):
    priorNodesSum = 0

    #use the grandparent nodes as a shortcut instead of doing a lot of complicated math
    grandparentNodes = hiddenVals[currentWeightLayer - 2]
    for j in range(len(grandparentNodes)):
        priorNodesSum += grandparentNodes[i] * weights[currentWeightLayer - 1][currentInputNode][j]

    return priorNodesSum

#gets the sum of the corresponding subsequent weights
# for the partial derivatives of weights in NN's with more than 1 weight set
def getSubsequentWeightsSum (weightSlopes, currentWeightLayer, currentResultantNode, currentOutputNode):
    subsequentWeightsSum = 0

    # loop through all subsequent weight sets
    for i in range(1, len(weightSlopes)):
        # iterate through all the lists/sets of weights in a subsequent weight set/layer (corresponding to resultant nodes)
        for j in range(len(weightSlopes[i])):
            # iterate through all the weights in each set
            for k in range(len(weightSlopes[i][j])):
                # weights[weight set/layer index][resultant node index][input node index]
                # if it's the 1st subsequent weight set
                if (i == currentWeightLayer + 1):
                    subsequentWeightsSum += weights[i][j][currentResultantNode]
                # if it's in any intermediate weight sets
                elif (i < len(weightSlopes) - 2):
                    subsequentWeightsSum += weights[i][j][k]
                # if it's in the last weight set
                else:
                    subsequentWeightsSum += weights[len(weightSlopes) - 1][currentOutputNode][currentResultantNode]

    return subsequentWeightsSum

def backpropagate (weights, slopes, bias):
    #make sure the arguments are right

    assert(len(weights) == numHiddenLayers + 1)
    assert(len(slopes) == numHiddenLayers + 1)
    assert(len(bias) == outputSize)

    #modify every weight by its corresponding slope based on the cost function
    #for each weight set
    for i in range(len(weights)):
        #for each array of weights corresponding to resultant node j
        for j in range(len(weights[i])):
            #for each weight in said array, which corresponds to each input node k
            for k in range(len(weights[i][j])):
                weights[i][j][k] -= stepSize * slopes[i][j][k]

    #modify the bias by its partial derivative
    for k in range(outputSize):
        bias[k] -= stepSize * 2 * (output[k] - proutput[k])

#assumes m1 and m2 are lists of the same length
def performFakeMatrixOperation (m1, operation, m2):
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

    if (numHiddenLayers > 0):
        hiddenNodeVals = np.zeros((hiddenLayerSize, numHiddenLayers))

        #define the first hidden layer
        for i in range(hiddenLayerSize):
            hiddenNodeVals[0][i] = sumWeightedValues(inputVals, weights[0][i])

        #define the rest of the hidden layers' nodes values
        if (numHiddenLayers > 1):
            #loop through the hidden layerrs
            for i in range(1, numHiddenLayers):
                #loop through the nodes in each layer
                for j in range(hiddenLayerSize):
                    #set the values of the nodes in the next hidden layer to the sum of the connected,
                    #weighted products of the previous layer's nodes
                    hiddenNodeVals[i][j] = sumWeightedValues(hiddenNodeVals[i - 1], weights[0][i])

    #calculate values of output nodes
    for i in range(outputSize):
        #calculate output
        if (numHiddenLayers > 0):
            output[i] = sumWeightedValues(hiddenNodeVals[numHiddenLayers - 1], weights[numHiddenLayers + 1][i]) + bias[i]
        else:
            output[i] = sumWeightedValues(inputVals, weights[0][i]) + bias[i]

        #calculate predicted output (target output)
        proutput[i] = getAvg(inputVals)

    #get the slopes for all weights
    #fix this so it works with no hidden layers
    if (numHiddenLayers > 0):
        weightSlopes = getCostSlopes(inputVals, hiddenNodeVals, output, proutput)
    else:
        weightSlopes = getCostSlopes(inputVals, None, output, proutput)

    #backpropagate
    backpropagate(weights, weightSlopes, bias)

    #Print stuff every few iterations
    if (iterationNum % 100 == 0):
        print("Running another round of training... (Iteration " + str(iterationNum) + ")\n"
              "******************************************************")

        # Print input values
        print("Input values: " + str(inputVals))
        print("Original weights: \n\t\t" + OGWeights)
        print("Weights (First " + str(inputSize * hiddenLayerSize) + " values are for the weights of the connections between the first 2 layers):\n\t\t" + str(weights))
        print("Bias: " + str(bias))

        # print output/backpropagation info
        print("Expected output: " + str(getAvg(inputVals)))
        print("Actual output: " + str(output))
        print("Difference: " + str(performFakeMatrixOperation(proutput, "-", output)))

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
            inputValues[j] = 1000   #random.randint(0, maxInValue)

        train(i, inputValues)
"""
Tutorials and sources used to build this:
************************************************************************************************************************
    This guy's tutorials are how I learned to build the perceptron:     https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
    This helped with the weird, dumb stuff with arrays:                 https://stackoverflow.com/questions/72585703/modifying-2d-array-python
    
"""
