import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """

        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        return ((x - mean) / (var + self.eps)**0.5) * self.gamma + self.beta
        

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    conv = x @ W1
    bn = bn1.forward(conv)
    rlu = relu(bn)
    conv = rlu @ W2
    bn = bn2.forward(conv)

    return relu(bn)
    
    
    

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """

    bn = bn1.forward(x)
    rlu = relu(bn)
    conv = rlu @ W1
    bn = bn1.forward(conv)
    rlu = relu(bn)
    conv = rlu @ W2

    return  conv
