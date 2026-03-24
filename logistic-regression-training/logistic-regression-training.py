import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).

    1. forward pass
    2. calculate loss
    3. backward pass
    4. update parameters
    
    """

    w_shape = X.shape[1]
    y_shape = y.shape
    N = y_shape[0]

    W = np.zeros(w_shape)
    b = 0.0

    for _ in range(steps):
        
        p = _sigmoid(X @ W + b)
    
        l = _binary_cross_entropy_loss(p, y, N)

        d_w = np.divide(np.transpose(X) @ (p - y), N)
        d_b = np.mean(p-y)

        W = W - lr * d_w
        b = b - lr * d_b

    return (W, b)
        

def _binary_cross_entropy_loss(p, y, N):

    loss = y * np.log(p) + ((1 - y) * np.log(1 - p))

    return np.sum(loss) / (-1 * N)