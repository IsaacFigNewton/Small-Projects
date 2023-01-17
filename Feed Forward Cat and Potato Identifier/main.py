import math
import random
import numpy as np
import matplotlib.pyplot as mp
import activation_functions as af

#To do:
#   Create void functions to set layer sizes more easily
#   Implement matplotlib or something to graph average loss each epoch


#Important variables
#***********************************************************************************************************************

epochCount = 25
trainingSetSize = 100
# the number of nodes in each layer (1st and last are I/O layers)
layerNodeCounts = [5, 4, 4, 3]
# the number of output nodes
learningRate = 10**3

#input vector; 100 sets of 5 input values (age (years), length (cm), width (cm), height (cm), mass (g))
#as well as the projected outputs for each layer
A = [np.zeros((layerNodeCounts[0], trainingSetSize)),
     np.zeros((layerNodeCounts[1], trainingSetSize)),
     np.zeros((layerNodeCounts[2], trainingSetSize)),
     np.zeros((layerNodeCounts[3], trainingSetSize))]

#weight vectors
W = [np.random.randint(-1000, 1000, (layerNodeCounts[0], layerNodeCounts[1])) / 1000,
     np.random.randint(-1000, 1000, (layerNodeCounts[1], layerNodeCounts[2])) / 1000,
     np.random.randint(-1000, 1000, (layerNodeCounts[2], layerNodeCounts[3])) / 1000]
dW = [np.zeros((layerNodeCounts[0], layerNodeCounts[1])),
      np.zeros((layerNodeCounts[1], layerNodeCounts[2])),
      np.zeros((layerNodeCounts[2], layerNodeCounts[3]))]

#biases
B = [np.zeros((layerNodeCounts[1], 1)),
     np.zeros((layerNodeCounts[2], 1)),
     np.zeros((layerNodeCounts[3], 1))]
dB = [np.zeros((layerNodeCounts[1], 1)),
      np.zeros((layerNodeCounts[2], 1)),
      np.zeros((layerNodeCounts[3], 1))]

#output vectors
# the pre-activation function outputs for each layer
Z = [np.zeros((layerNodeCounts[1], trainingSetSize)),
     np.zeros((layerNodeCounts[2], trainingSetSize)),
     np.zeros((layerNodeCounts[3], trainingSetSize))]
dZ = [np.zeros((layerNodeCounts[1], trainingSetSize)),
      np.zeros((layerNodeCounts[2], trainingSetSize)),
      np.zeros((layerNodeCounts[3], trainingSetSize))]

# the desired output for the last A layer
Y = np.zeros((trainingSetSize, layerNodeCounts[-1]))


#Set the input and output
#***********************************************************************************************************************

def setIOVals():
    #choose which output should be "turned on", i.e. what the output should be for each training set
    yIndices = np.random.randint(0, 3, size=(trainingSetSize, 1))

    #set those output nodes' values to their "on" states
#    np.put_along_axis(y, yIndices, 1, axis=0) #possibly faster
    for i in range(0, trainingSetSize):
        # print(yIndices[i])
        #                                             indices should be swapped but the IDE gets mad when I do so
        Y[i][yIndices[i]] = 1

        #if it's not a cat or a potato
        if (yIndices[i] == 0):
            #set random age
            A[0][0][i] = random.uniform(10**-3, 10**5)
            #set random length
            A[0][1][i] = random.uniform(10**-3, 10**5)
            #set random width
            A[0][2][i] = random.uniform(10**-3, 10**5)
            #set random height
            A[0][3][i] = random.uniform(10**-3, 10**5)
            #set random mass
            A[0][4][i] = random.uniform(10**-3, 10**2)

        # if it's a cat
        if (yIndices[i] == 1):
            # set random age
            A[0][0][i] = random.uniform(10 ** -3, 27)
            # based on an average adulthood age of 8.5yrs +- 1.5yrs
            growthVariability = np.random.normal(0, 0.176)
            #Desmos very good
            amtGrwthCmpltd = af.sigmoid((A[0][0][i] - (0.7 + growthVariability) * 5) / (0.7 + growthVariability))

            # set random length
            A[0][1][i] = abs(np.random.normal(8, 2) + amtGrwthCmpltd * np.random.normal(38, 5))
            # set random width
            A[0][2][i] = abs(np.random.normal(3, 1) + amtGrwthCmpltd * np.random.normal(10, 4))
            # set random height
            A[0][3][i] = abs(np.random.normal(0.5, 0.1) + amtGrwthCmpltd * np.random.normal(25, 2.5))
            # set random mass
            A[0][4][i] = abs(np.random.normal(114, 28) + amtGrwthCmpltd * (np.random.normal(4900, 1500) + 1000))

            # print("\nCat stats: ", X[i], "\n")

        # if it's a potato
        if (yIndices[i] == 2):
            # set random age
            maxAge = np.random.normal(1.0 / 6, 1.0 / 12)
            A[0][0][i] = random.uniform(10 ** -3, maxAge)
            # based on yukon gold potatoes' an average adulthood age of 125 days +- 15 days
            growthVariability = np.random.normal(0, 0.0033)
            amtGrwthCmpltd = af.sigmoid((A[0][0][i] - (0.028 + growthVariability) * 5) / (0.028 + growthVariability))

            # set semi-random length
            A[0][1][i] = abs(amtGrwthCmpltd * np.random.normal(6.5, 2.5))
            # set semi-random width
            A[0][2][i] = abs(A[0][1][i] + amtGrwthCmpltd * np.random.normal(0, 4))
            # set semi-random height
            A[0][3][i] = abs(A[0][1][i] + amtGrwthCmpltd * np.random.normal(0, 2.5))
            # mass = volume of ellipsoid potato * 1.08g/cm^3
            A[0][4][i] = abs(4.0/3 * math.pi * A[0][1][i] * A[0][2][i] * A[0][3][i] * 1.08)

            # print("\nPotato stats: ", X[i], "\n")

    # print("\nY after setting output node states:\n", Y)


# Important functions here
# **********************************************************************************************************************
# set the activation function for hidden layers
def g(x):
    return af.ReLU(x)

def dg(x):
    return af.dReLU(x)

def getLoss(a, y):
    # try:
    #     print(a)
        return -(y * np.log(a) + (1 - y) * np.log(1 - a + 10**-10))
    # except RuntimeWarning:
    #     return float('nan')

def getCost(loss, m):
    return np.sum(loss)/m

def dz(a, y):
    return a - y


#Main function here
#***********************************************************************************************************************
if __name__ == "__main__":
    averageLosses = []

    for i in range(epochCount):
        # propagate forward
        # *************************************************************************************************************
        setIOVals()

        assert(np.shape(B[1]) == (layerNodeCounts[2], 1))

        #get predicted outputs for each hidden and output layer
        for i in range(len(Z)):
            Z[i] = np.dot(np.transpose(W[i]), A[i]) + B[i]
            A[i + 1] = af.vectorizedSigmoid(Z[i])
            # print("\nA:\n", A)

            assert(np.shape(A[i + 1]) == (layerNodeCounts[i + 1], trainingSetSize))


        # backpropagate
        # *************************************************************************************************************
        #get dz in last layer
        dZ[-1] = A[-1] - Y.transpose()
        
        for i in range(len(layerNodeCounts) - 2, -1, -1):
            dW[i] = np.dot(A[i], dZ[i].transpose()) / trainingSetSize
            assert(np.shape(dW[i]) == (layerNodeCounts[i], layerNodeCounts[i + 1]))

            # add up the values in each column
            dB[i] = np.sum(dZ[i], axis=1) / trainingSetSize
            # gotta reshape db's because they've got shapes of (layerNodeCounts[2],)
            dB[i] = np.reshape(dB[i], (layerNodeCounts[i + 1], 1))

            if (i > 0):
                dZ[i - 1] = np.dot(W[i], dZ[i]) * dg(Z[i - 1])


        # descend the gradient
        for i in range(len(W)):
            W[i] = W[i] - learningRate * dW[i]
            # W[1] = W[1] - learningRate * dW[1]
            # B[0] = B[0] - learningRate * dB[0]
            B[i] = B[i] - learningRate * dB[i]

        # print("\nW[0]:\n", W[0])
        # print("\nW[1]:\n", W[1])
        cost = getCost(getLoss(A[-1], Y.transpose()), trainingSetSize)
        averageLosses.append(cost)

        print("\nAverage Loss: ", cost)

    mp.plot([i for i in range(epochCount)], averageLosses)
    # mp.xscale("log")
    mp.xlabel("Epoch Number")
    mp.ylabel("Average Loss")
    mp.show()

#Sauces:
#Most stuff                             https://www.coursera.org/learn/neural-networks-deep-learning/
#NumPy Docs                             https://numpy.org/doc/stable/index.html
#Desmos for input functions             https://www.desmos.com/calculator