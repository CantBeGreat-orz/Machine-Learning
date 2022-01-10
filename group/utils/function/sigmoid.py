import numpy as np

def sigmoid(x):
    tmp = np.exp(x)
    return tmp/(1+tmp)