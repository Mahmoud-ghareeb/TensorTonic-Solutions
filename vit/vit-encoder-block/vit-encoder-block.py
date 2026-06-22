import numpy as np

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    return x_norm

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def multi_head_self_attn(x, num_heads, Wq, Wk, Wv, Wo):

    b, s, d = x.shape

    dk = d // num_heads

    Q = (x @ Wq).reshape(b, s, num_heads, dk).transpose(0, 2, 1, 3)
    K = (x @ Wk).reshape(b, s, num_heads, dk).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(b, s, num_heads, dk).transpose(0, 2, 1, 3)

    attn = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dk), axis=-1) @ V

    return attn.transpose(0, 2, 1, 3).reshape(b, s, d) @ Wo


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def mlp(x, W1, W2):

    return gelu(x @ W1) @W2
    
    

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """

    hidden_dim = int(embed_dim * mlp_ratio)
    
    if Wq is None:
        Wq = np.random.randn(embed_dim, embed_dim) * 0.02

    if Wk is None:
        Wk = np.random.randn(embed_dim, embed_dim) * 0.02

    if Wv is None:
        Wv = np.random.randn(embed_dim, embed_dim) * 0.02

    if Wo is None:
        Wo = np.random.randn(embed_dim, embed_dim) * 0.02

    if W1 is None:
        W1 = np.random.randn(embed_dim, hidden_dim) * 0.02

    if W2 is None:
        W2 = np.random.randn(hidden_dim, embed_dim) * 0.02

    res = x
    x = layer_norm(x)
    x = multi_head_self_attn(x, num_heads, Wq, Wk, Wv, Wo)
    x += res 
    res = x
    x = layer_norm(x)
    x = mlp(x, W1, W2)

    return x + res
    
    
    