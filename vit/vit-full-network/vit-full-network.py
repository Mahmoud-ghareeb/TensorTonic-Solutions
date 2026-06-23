import numpy as np


def get_patch_embeddings(x, patch_size, num_patches, W_patch):
    """Split image into patches, flatten, and linearly project.
    x: (B, H, W, C) -> (B, num_patches, embed_dim)
    W_patch: (patch_size*patch_size*C, embed_dim)  -- passed in, NOT created here.
    """
    b, h, w, c = x.shape
    n = num_patches

    x = x.reshape(b, h // patch_size, patch_size, w // patch_size, patch_size, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(b, n, patch_size * patch_size * c)

    return x @ W_patch


def append_cls_token(x, cls_token):
    b, _, _ = x.shape
    cls_token = np.tile(cls_token, (b, 1, 1))
    return np.concatenate([cls_token, x], axis=1)


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


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
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def mlp(x, W1, W2):
    return gelu(x @ W1) @ W2


def vit_encoder_block(x, num_heads, Wq, Wk, Wv, Wo, W1, W2):
    """ViT encoder block with Pre-LayerNorm. All weights are passed in."""
    res = x
    x = layer_norm(x)
    x = multi_head_self_attn(x, num_heads, Wq, Wk, Wv, Wo)
    x = x + res

    res = x
    x = layer_norm(x)
    x = mlp(x, W1, W2)
    return x + res


def classification_head(encoder_output, W_head):
    """Take [CLS] token, LayerNorm, then project. (B, N, D) -> (B, num_classes)"""
    cls = encoder_output[:, 0, :]
    return layer_norm(cls) @ W_head


class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """Weights are initialized ONCE here and stored. If arrays are passed in,
        they are used as-is; otherwise initialized randomly."""
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.hidden_dim = int(embed_dim * mlp_ratio)

        patch_dim = patch_size * patch_size * 3

        # --- persistent weights ---
        self.W_patch = W_patch if W_patch is not None \
            else np.random.randn(patch_dim, embed_dim) * 0.02
        self.cls_token = cls_token if cls_token is not None \
            else np.random.randn(1, 1, embed_dim) * 0.02
        self.pos_embed = pos_embed if pos_embed is not None \
            else np.random.randn(1, self.num_patches + 1, embed_dim) * 0.02

        if encoder_weights is not None:
            self.encoder_weights = encoder_weights
        else:
            self.encoder_weights = []
            for _ in range(depth):
                self.encoder_weights.append({
                    "Wq": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wk": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wv": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wo": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "W1": np.random.randn(embed_dim, self.hidden_dim) * 0.02,
                    "W2": np.random.randn(self.hidden_dim, embed_dim) * 0.02,
                })

        self.W_head = W_head if W_head is not None \
            else np.random.randn(embed_dim, num_classes) * 0.02

    def forward(self, x):
        b, h, w, c = x.shape

        x = get_patch_embeddings(x, self.patch_size, self.num_patches, self.W_patch)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for blk in self.encoder_weights:
            x = vit_encoder_block(
                x, self.num_heads,
                blk["Wq"], blk["Wk"], blk["Wv"], blk["Wo"], blk["W1"], blk["W2"],
            )

        return classification_head(x, self.W_head)
