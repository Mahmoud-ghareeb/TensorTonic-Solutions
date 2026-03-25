import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """

    b, s, dk = Q.shape
    
    QK = Q @ K.permute(0, 2, 1)
    
    SftMX = F.softmax(QK * (dk**-0.5), dim=-1)

    Attn = SftMX @ V

    return Attn