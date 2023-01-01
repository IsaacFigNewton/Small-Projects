import math
import random
import numpy as np

#To do:
#   Fix sigmoid() and getLoss so you don't have to compensate for divide-by-zero errors


#Important vars here
#***********************************************************************************************************************

epochCount = 10000
trainingSetSize = 100
inCount = 5
outCount = 3
learningRate = 10**-5

#input vector; 100 sets of 5 input values (age (years), length (cm), width (cm), height (cm), mass (g))
X = np.zeros((trainingSetSize, inCount))
#weight vector; 5 input x 3 output
W = np.random.randint(-1000, 1000, (inCount, outCount)) / 1000
dw = np.zeros((inCount, outCount))
#biases
B = np.zeros((outCount, 1))
db = np.zeros((outCount, 1))
#output vectors
A = np.zeros((trainingSetSize, outCount))
Y = np.zeros((trainingSetSize, outCount))
dz = np.zeros((trainingSetSize, outCount))



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
    yIndices = np.random.randint(0, 3, size=(100, 1))

    #set those output nodes' values to their "on" states
#    np.put_along_axis(y, yIndices, 1, axis=0) #possibly faster
    for i in range(0, trainingSetSize):
        Y[i][yIndices[i]] = 1

        #if it's not a cat or a potato
        if (yIndices[i] == 0):
            #set random age
            X[i][0] = random.uniform(10**-3, 10**5)
            #set random length
            X[i][1] = random.uniform(10**-3, 10**5)
            #set random width
            X[i][2] = random.uniform(10**-3, 10**5)
            #set random height
            X[i][3] = random.uniform(10**-3, 10**5)
            #set random mass
            X[i][4] = random.uniform(10**-3, 10**2)

        # if it's a cat
        if (yIndices[i] == 1):
            # set random age
            X[i][0] = random.uniform(10 ** -3, 27)
            # based on an average adulthood age of 8.5yrs +- 1.5yrs
            growthVariability = np.random.normal(0, 0.176)
            amtGrwthCmpltd = sigmoid((X[i][0] - (0.7 + growthVariability) * 5) / (0.7 + growthVariability))

            # set random length
            X[i][1] = abs(np.random.normal(8, 2) + amtGrwthCmpltd * np.random.normal(38, 5))
            # set random width
            X[i][2] = abs(np.random.normal(3, 1) + amtGrwthCmpltd * np.random.normal(10, 4))
            # set random height
            X[i][3] = abs(np.random.normal(0.5, 0.1) + amtGrwthCmpltd * np.random.normal(25, 2.5))
            # set random mass
            X[i][4] = abs(np.random.normal(114, 28) + amtGrwthCmpltd * (np.random.normal(4900, 1500) + 1000))

            # print("\nCat stats: ", X[i], "\n")

        # if it's a potato
        if (yIndices[i] == 2):
            # set random age
            maxAge = np.random.normal(1.0 / 6, 1.0 / 12)
            X[i][0] = random.uniform(10 ** -3, maxAge)
            # based on yukon gold potatoes' an average adulthood age of 125 days +- 15 days
            growthVariability = np.random.normal(0, 0.0033)
            amtGrwthCmpltd = sigmoid((X[i][0] - (0.028 + growthVariability) * 5) / (0.028 + growthVariability))

            # set semi-random length
            X[i][1] = abs(amtGrwthCmpltd * np.random.normal(6.5, 2.5))
            # set semi-random width
            X[i][2] = abs(X[i][1] + amtGrwthCmpltd * np.random.normal(0, 4))
            # set semi-random height
            X[i][3] = abs(X[i][1] + amtGrwthCmpltd * np.random.normal(0, 2.5))
            # mass = volume of ellipsoid potato * 1.08g/cm^3
            X[i][4] = abs(4.0/3 * math.pi * X[i][1] * X[i][2] * X[i][3] * 1.08)

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
        setIOVals()

        #get predicted outputs
        A = vectorizedSigmoid(np.dot(X, W) + np.transpose(B))
        # print("\nA:\n", A)

        #get dz
        dz = A - Y

        #find costs and set dw and db appropriately
        #may have messed something up in the variables' config because I'm transposing X here
        dw = np.dot(np.transpose(X), dz) / trainingSetSize
        #add up the values in each column
        db = np.sum(dz, axis=0) / trainingSetSize
        db = np.reshape(db, (3, 1))

        # print("\ndw:\n", dw)
        assert(np.shape(dw) == (inCount, outCount))
        # print("\ndb:\n", db)
        assert(np.shape(db) == (outCount, 1))

        W -= learningRate * dw
        B -= learningRate * db

        print("\nAverage Loss: ", getCost(getLoss(A, Y), trainingSetSize))

#Sauces:
#Pretty much everything                 https://www.coursera.org/learn/neural-networks-deep-learning/
#NumPy Docs                             https://numpy.org/doc/stable/index.html
#Desmos for input functions             https://www.desmos.com/calculator