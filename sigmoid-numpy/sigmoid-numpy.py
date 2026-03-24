import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here

    return (np.divide(1, np.add(1, np.exp(np.multiply(-1, x)))))