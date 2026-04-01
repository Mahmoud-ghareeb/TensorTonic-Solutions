import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """

    g = np.eye(gradients_F[0].shape[0])
    for l in gradients_F:
        g = g @ (l + np.eye(gradients_F[0].shape[0]))
        
    return x @ g

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    g = np.eye(gradients_F[0].shape[0])
    for l in gradients_F:
        g = g @ l
        
    return x @ g
