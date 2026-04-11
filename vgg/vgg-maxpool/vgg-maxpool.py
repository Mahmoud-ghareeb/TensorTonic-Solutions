import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    B, H, W, C = x.shape

    H_out = (H - 2) // 2 + 1
    W_out = (W - 2) // 2 + 1

    out = np.zeros((B, H_out, W_out, C))

    for b in range(B):
        for i in range(H_out):
            for j in range(W_out):
                for c in range(C):
                    window = x[b, i*2:i*2+2, j*2:j*2+2, c]
                    out[b, i, j, c] = np.max(window)

    return out