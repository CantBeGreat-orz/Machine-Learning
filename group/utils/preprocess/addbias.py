import numpy as np

def add_bias(x):
    '''add bias'''
    return np.hstack((np.ones((x.shape[0], 1)), x))