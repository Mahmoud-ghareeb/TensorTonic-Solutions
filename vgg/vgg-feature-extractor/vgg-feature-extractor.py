import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    i = 0
    for c in config:
        if c == 'M':
            x = maxpool_2x2(x)
            continue

        x = x @conv_weights[i] + conv_biases[i]
        x = np.maximum(0, x)
        i += 1

    return x