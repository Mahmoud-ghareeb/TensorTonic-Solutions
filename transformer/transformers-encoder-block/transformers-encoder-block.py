import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return gamma * ((x - mean) / (var + eps)**0.5) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    b, s, d = Q.shape
    dk = d // num_heads

    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v
    
    Q = np.transpose(Q.reshape(b, s, num_heads, dk), (0, 2, 1, 3))
    K = np.transpose(K.reshape(b, s, num_heads, dk), (0, 2, 1, 3))
    V = np.transpose(V.reshape(b, s, num_heads, dk), (0, 2, 1, 3))

    attn = softmax((Q @ np.transpose(K, (0, 1, 3, 2))) * dk**-0.5) @ V

    return np.transpose(attn, (0, 2, 1, 3)).reshape(b, s, d) @ W_o
    
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    
    h1 = x @ W1 + b1
    y1 = np.maximum(0, h1)

    return y1 @ W2 + b2
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    
    x = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)

    return layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)