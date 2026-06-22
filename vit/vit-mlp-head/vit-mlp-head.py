import numpy as np

def layer_norm(x, eps=1e-6):

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    return (x - mean) / (np.sqrt(var) + eps)

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """

    b, n, d = encoder_output.shape
    
    if W_head is None:
        W_head = np.random.randn(d, num_classes) * 0.02

    return layer_norm(encoder_output[:, 0, :]) @ W_head