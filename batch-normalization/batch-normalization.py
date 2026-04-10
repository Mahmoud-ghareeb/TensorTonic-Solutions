import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    if x.ndim == 2:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        
    elif x.ndim == 4:
        N, C, H, W = x.shape
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        gamma = np.array(gamma).reshape(1, C, 1, 1)
        beta = np.array(beta).reshape(1, C, 1, 1)

    return gamma * ((x - mean) / (var + eps)**0.5) + beta