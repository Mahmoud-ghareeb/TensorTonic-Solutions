import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    B, _, _ = patches.shape

    if cls_token is None:
        cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        
    cls_token = np.tile(cls_token, (B, 1, 1))
    
    return np.concatenate([cls_token, patches], axis=1)
    