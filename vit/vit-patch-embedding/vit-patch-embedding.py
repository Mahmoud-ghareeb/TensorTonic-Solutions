import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """

    b, h, w, c = image.shape
    
    n = (h//patch_size) * (w//patch_size)

    x = image.reshape(b, h//patch_size, patch_size, w//patch_size, patch_size, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(b, n, patch_size*patch_size*c)

    if W_proj is None:                          
        W_proj = np.random.randn(patch_size*patch_size*c, embed_dim) * 0.02

    return x @ W_proj