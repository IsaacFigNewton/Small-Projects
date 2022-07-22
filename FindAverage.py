#A simple neural network that learns to find the average of some numbers
import random

#most important vars
#use input size of 2 for a perceptron
inputSize = 10
stepSize = 10 ** -11
iterations = 1000

maxInValue = 10 ** 5

#instantiate the input list/input nodes
inputValues = [0] * inputSize

#instantiate and define the corresponding weights (range from -1 to 1)
inputWeights = [0.0] * inputSize
for i in range(inputSize):
    inputWeights[i] = random.randint(-1000, 1000)/1000.0
#for debugging
OGWeights = str(inputWeights)

bias = 0
proutput = 0
output = 0

#set guiding function to train perceptron 2
def getAvg (inputVals):
    #add up all inputs
    sum = 0
    for val in inputVals:
        sum += val

    #get the average
    return float(sum) / inputSize

def getOutput(inputVals, inputWeights):
    # multiply by weights
    weightedInputs = [0] * inputSize
    for i in range(inputSize):
        weightedInputs[i] = inputVals[i] * inputWeights[i]

    # sum weighted inputs and add bias for output node
    output = 0
    for val in weightedInputs:
        output += val
    output += bias
    return output

#cost function (square error)
def getCost(output, proutput):
    return (output - proutput) ** 2

def getCostSlopes(inputVals, output, proutput):
    #define a list to store the slopes at the weights in
    weightSlopes = [0] * inputSize

    #calculate the partial derivatives for all weights with respect to each weight
    # cost = (output - proutput) ** 2
    # output = in1 * w1 + in2 * w2 + ... + b
    # d/dw1(output) = in1
    # thus d/dw1(cost) = 2 * (output - proutput) * in1
    # and d/dw...(cost) = 2 * (output - proutput) * in...

    for i in range(inputSize):
        #...and store it in a list of weights' corresponding slopes based on the cost function
        weightSlopes[i] = 2 * (output - proutput) * inputVals[i]

    return weightSlopes

def backpropagate (inputWeights, slopes, bias):
    #modify every weight by its corresponding slope based on the cost function
    for i in range(inputSize):
        inputWeights[i] -= stepSize * slopes[i]

    #modify the bias by its partial derivative
    bias -= stepSize * 2 * (output - proutput)


#define the training procedure for 1 round of training
def train (iterationNum, inputVals):
    #calculate output
    output = getOutput(inputVals, inputWeights)

    #calculate predicted output (target output)
    proutput = getAvg(inputVals)


    weightSlopes = getCostSlopes(inputVals, output, proutput)

    #backpropagate
    backpropagate(inputWeights, weightSlopes, bias)

    #Print stuff every few iterations
    if (iterationNum % 100 == 0):
        print("Running another round of training... (Iteration " + str(iterationNum) + ")\n"
              "******************************************************")

        # Print input values
        print("Input values: " + str(inputVals))
        print("Original weights: " + OGWeights)
        print("Weights: " + str(inputWeights))
        print("Bias: " + str(bias))

        # print output/backpropagation info
        print("Expected output: " + str(getAvg(inputVals)))
        print("Actual output: " + str(output))
        print("Difference: " + str(proutput - output))

        print ("Backpropagating...")
        print("Weight slopes: " + str(weightSlopes))

        #print an empty line to make the debugging look nice :)
        print("")

#run the NN
if __name__ == '__main__':
    print('Starting program...')

    print("Training...")
    for i in range(iterations):

        #define the input values
        for j in range(0, inputSize):
            inputValues[j] = random.randint(0, maxInValue)

        train(i, inputValues)

#Tutorials and sources used to build this:
#   This guy's tutorials are how I learned to build the perceptron: https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
