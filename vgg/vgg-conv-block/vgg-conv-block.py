import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """

    _, _, _, c = x.shape
    w1 = np.random.randn(c, out_channels)
    w2 = np.random.randn(out_channels, out_channels)
    b = np.random.randn(out_channels)
    
    for _ in range(num_convs):

        if _ == 0:
            x = np.maximum(0, x @ w1 + b)
        else:
            x = np.maximum(0, x @ w2 + b)

    return x