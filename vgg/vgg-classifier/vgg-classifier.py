import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    
    b, h, w, c = features.shape
    x = features.reshape(b, -1)
    x = x@W1 + b1
    x = np.maximum(0, x)
    x = x @ W2 + b2
    x = np.maximum(0, x)
    x = x @ W3 + b3

    return x
    