import numpy as np

def min_max_normalization(x):
    '''normalization'''
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm
