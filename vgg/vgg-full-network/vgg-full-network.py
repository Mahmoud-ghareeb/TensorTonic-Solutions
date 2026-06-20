import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x, config, conv_weights, conv_biases):
    out = x.copy()
    w_idx = 0
    for layer in config:
        if layer == 'M':
            out = maxpool_2x2(out)
        else:
            out = out @ conv_weights[w_idx] + conv_biases[w_idx]
            out = np.maximum(0, out)
            w_idx += 1
    return out

def vgg_classifier(features, W1, b1, W2, b2, W3, b3):
    B = features.shape[0]
    x = features.reshape(B, -1)
    x = np.maximum(0, x @ W1 + b1)
    x = np.maximum(0, x @ W2 + b2)
    return x @ W3 + b3

def vgg16(x: np.ndarray, config: list, conv_weights: list, conv_biases: list,
          W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
          W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    
    x = vgg_features(x, config, conv_weights, conv_biases)
    return vgg_classifier(x, W1, b1, W2, b2, W3, b3)