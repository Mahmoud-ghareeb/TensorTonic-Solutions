import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    b, s, d = Q.shape

    dk = d // num_heads
    
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    Q = np.transpose(Q.reshape(b, s, num_heads, dk), (0, 2, 1, 3))
    K = np.transpose(K.reshape(b, s, num_heads, dk), (0, 2, 1, 3))
    V = np.transpose(V.reshape(b, s, num_heads, dk), (0, 2, 1, 3))

    QK = Q @ np.transpose(K, (0, 1, 3, 2))
    sftmx = softmax(QK * (dk ** -0.5))

    attn = np.transpose(sftmx @ V, (0, 2, 1, 3))

    attn = attn.reshape(b, s, d)

    return attn @ W_o

    