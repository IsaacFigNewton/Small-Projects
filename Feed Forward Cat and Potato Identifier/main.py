import math
import random
import numpy as np

#To do:
#   Fix sigmoid() and getLoss so you don't have to compensate for divide-by-zero errors
#   Fix partial derivatives for weight and bias layers
#   Find out why you keep getting "IndexError: index 1 is out of bounds for axis 0 with size 1" for line 74
#       when you try to correct it



#Important vars here
#***********************************************************************************************************************

epochCount = 20
trainingSetSize = 1000
inCount = 5
# represents the number of hidden layers + the input layer
# numLayers = 2
# the number of hidden layer nodes
hiddenCount = 4
# the number of output nodes
outCount = 3
learningRate = 10**-6

#input vector; 100 sets of 5 input values (age (years), length (cm), width (cm), height (cm), mass (g))
A0 = np.zeros((inCount, trainingSetSize))
#weight vector; 5 input x 3 output
W0 = np.random.randint(-1000, 1000, (inCount, hiddenCount)) / 1000
W1 = np.random.randint(-1000, 1000, (hiddenCount, outCount)) / 1000
dw0 = np.zeros((inCount, hiddenCount))
dw1 = np.zeros((hiddenCount, outCount))
#biases
B0 = np.zeros((hiddenCount, 1))
B1 = np.zeros((outCount, 1))
db0 = np.zeros((hiddenCount, 1))
db1 = np.zeros((outCount, 1))
#output vectors
# the projected outputs for each layer
A1 = np.zeros((hiddenCount, trainingSetSize))
A2 = np.zeros((outCount, trainingSetSize))
# the desired output for the last A layer
#                                                                        dimensions should be in reverse order
Y = np.zeros((trainingSetSize, outCount))
dz = np.zeros((outCount, trainingSetSize))



#Important functions here
#***********************************************************************************************************************

def sigmoid(x):
    # try:
    if (x > 100):
        return 1 - 10**-10
    elif (x < -100):
        return 0 + 10**-10
    else:
        return 1.0 / (1 + math.exp(-x))

    # except OverflowError:
    #     return float('inf')

vectorizedSigmoid = np.vectorize(sigmoid)

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
            A0[0][i] = random.uniform(10**-3, 10**5)
            #set random length
            A0[1][i] = random.uniform(10**-3, 10**5)
            #set random width
            A0[2][i] = random.uniform(10**-3, 10**5)
            #set random height
            A0[3][i] = random.uniform(10**-3, 10**5)
            #set random mass
            A0[4][i] = random.uniform(10**-3, 10**2)

        # if it's a cat
        if (yIndices[i] == 1):
            # set random age
            A0[0][i] = random.uniform(10 ** -3, 27)
            # based on an average adulthood age of 8.5yrs +- 1.5yrs
            growthVariability = np.random.normal(0, 0.176)
            #Desmos very good
            amtGrwthCmpltd = sigmoid((A0[0][i] - (0.7 + growthVariability) * 5) / (0.7 + growthVariability))

            # set random length
            A0[1][i] = abs(np.random.normal(8, 2) + amtGrwthCmpltd * np.random.normal(38, 5))
            # set random width
            A0[2][i] = abs(np.random.normal(3, 1) + amtGrwthCmpltd * np.random.normal(10, 4))
            # set random height
            A0[3][i] = abs(np.random.normal(0.5, 0.1) + amtGrwthCmpltd * np.random.normal(25, 2.5))
            # set random mass
            A0[4][i] = abs(np.random.normal(114, 28) + amtGrwthCmpltd * (np.random.normal(4900, 1500) + 1000))

            # print("\nCat stats: ", X[i], "\n")

        # if it's a potato
        if (yIndices[i] == 2):
            # set random age
            maxAge = np.random.normal(1.0 / 6, 1.0 / 12)
            A0[0][i] = random.uniform(10 ** -3, maxAge)
            # based on yukon gold potatoes' an average adulthood age of 125 days +- 15 days
            growthVariability = np.random.normal(0, 0.0033)
            amtGrwthCmpltd = sigmoid((A0[0][i] - (0.028 + growthVariability) * 5) / (0.028 + growthVariability))

            # set semi-random length
            A0[1][i] = abs(amtGrwthCmpltd * np.random.normal(6.5, 2.5))
            # set semi-random width
            A0[2][i] = abs(A0[1][i] + amtGrwthCmpltd * np.random.normal(0, 4))
            # set semi-random height
            A0[3][i] = abs(A0[1][i] + amtGrwthCmpltd * np.random.normal(0, 2.5))
            # mass = volume of ellipsoid potato * 1.08g/cm^3
            A0[4][i] = abs(4.0/3 * math.pi * A0[1][i] * A0[2][i] * A0[3][i] * 1.08)

            # print("\nPotato stats: ", X[i], "\n")

    # print("\nY after setting output node states:\n", Y)



def getLoss(a, y):
    try:
        # print(a)
        return -(y * np.log(a) + (1 - y) * np.log(1 - a + 10**-10))
    except RuntimeWarning:
        return float('nan')

def getCost(loss, m):
    return np.sum(loss)/m

def dz(a, y):
    return a - y



#Main function here
#***********************************************************************************************************************
if __name__ == "__main__":
    for i in range(epochCount):
        # propagate forward
        # *************************************************************************************************************
        setIOVals()


        assert(np.shape(B1) == (outCount, 1))

        #get predicted outputs
        A1 = vectorizedSigmoid(np.dot(np.transpose(W0), A0) + B0)
        A2 = vectorizedSigmoid(np.dot(np.transpose(W1), A1) + B1)
        # print("\nA:\n", A)

        assert(np.shape(A1) == (hiddenCount, trainingSetSize))
        assert(np.shape(A2) == (outCount, trainingSetSize))

        # backpropagate
        # *************************************************************************************************************
        #get dz
        #                                                           shouldn't need to do y.transpose()
        dz = A2 - Y.transpose()

        #find costs and set dw's and db's appropriately
        #                                                   when you fix y.transpose(), you won't need dz.transpose here
        dw1 = np.dot(A1, dz.transpose()) / trainingSetSize
        assert(np.shape(dw0) == (inCount, hiddenCount))
        dw1 = np.dot(A1, dz.transpose()) / trainingSetSize
        assert(np.shape(dw1) == (hiddenCount, outCount))

        #add up the values in each column
        # db0 = np.sum(dz, axis=1) / trainingSetSize
        db1 = np.sum(dz, axis=1) / trainingSetSize
        # gotta reshape db's because they've got shapes of (outCount,)
        # db0 = np.reshape(db0, (hiddenCount, 1))
        db1 = np.reshape(db1, (outCount, 1))


        W0 = W0 - learningRate * dw0
        W1 = W1 - learningRate * dw1
        B0 = B0 - learningRate * db0
        B1 = B1 - learningRate * db1

        print("\nAverage Loss: ", getCost(getLoss(A2, Y.transpose()), trainingSetSize))

#Sauces:
#Pretty much everything                 https://www.coursera.org/learn/neural-networks-deep-learning/
#NumPy Docs                             https://numpy.org/doc/stable/index.html
#Desmos for input functions             https://www.desmos.com/calculator