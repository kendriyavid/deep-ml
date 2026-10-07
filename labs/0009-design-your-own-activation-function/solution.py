import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.

    Args:
        x: numpy array of any shape (raw neuron outputs)

    Returns:
        numpy array of same shape (activated outputs)

    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # np.maximum creates and returns a brand new array, 
    # comparing each element to 0 and keeping the higher value.
    return np.maximum(0, x)