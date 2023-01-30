import math
import numpy as np

def sigmoid(x):
    # try:
    if (x < -100):
        return 1 - 10**-10
    # elif (x < -100):
    #     return 0 + 10**-10
    else:
        return 1.0 / (1 + math.exp(-x))

    # except OverflowError:
    #     return float('inf')


def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    # try:
    if (x < -100):
        return 1 - 10**-10
    # elif (x < -100):
    #     return 0 + 10**-10
    else:
        return 1.0 / (1 + math.exp(-x))

    # except OverflowError:
    #     return float('inf')


def dTanh(x):
    return 1 - (tanh(x)) ** 2


def ReLU(x):
    return np.max(0, x)


def dReLU(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

vectorizedSigmoid = np.vectorize(sigmoid)
vectorizedDSigmoid = np.vectorize(dSigmoid)
vectorizedTanh = np.vectorize(tanh)
vectorizedDTanh = np.vectorize(dTanh)
vectorizedReLU = np.vectorize(ReLU)
vectorizedDReLU = np.vectorize(dReLU)