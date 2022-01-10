import numpy as np

def max_predict(probability, shape):
    return np.argmax(probability, axis=1).reshape(shape)