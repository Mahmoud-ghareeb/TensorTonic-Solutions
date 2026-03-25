import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    pe = np.zeros((seq_length, d_model))
    pos = np.arange(seq_length).reshape(-1, 1)
    i = np.arange(0, d_model, 2)
    div = pos * np.exp(2 * i * (-1 * np.log(10000) / d_model))

    pe[:, 0::2] = np.sin(div)
    pe[:, 1::2] = np.cos(div)

    return pe
    
    