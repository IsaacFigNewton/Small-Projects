#A simple neural network that learns to find the average of some numbers
import random

inputSize = 2

#instantiate the input list
inputValues = [0] * inputSize
#instantiate the corresponding weights
inputWeights = [0] * inputSize
#instantiate the bias
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


def backpropagate (inputVals, inputWeights, cost):
    print ("Backpropagating...")


#define the training procedure for 1 round of training
def train (inputVals):
    print("Running another round of training...\n"
          "******************************************************")


    #Format and print input values
    inputString = "["
    for val in inputVals:
        inputString += str(val) + ", "
    inputString = inputString[0:-2] + "]"

    print("Input values:")
    print(inputString)


    #multiply by weights
    weightedInputs = [0] * inputSize
    for i in range(0, inputSize):
        weightedInputs[i] = inputVals[i] * inputWeights[i]


    #sum weighted inputs and add bias for output node
    output = 0
    for val in weightedInputs:
        output += val

    output += bias
    proutput = getAvg(inputVals)


    #compare/print expected and actual outputs
    print("Expected output:" + str(getAvg(inputVals)))
    print("Actual output:" + str(output))
    print("Difference:" + str(proutput - output))


    #determine cost
    cost = (proutput - output) ** 2


    #backpropagate
    backpropagate(inputVals, inputWeights, cost)


    #print an empty line to make the debugging look nice :)
    print("")


if __name__ == '__main__':
    print('Starting program...')
    print("Training...")

    for i in range(0, 1000):
        #define the input values
        for val in inputValues:
            val = random.randint(0, 1000)

        train(inputValues)