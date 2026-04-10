import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if seed is not None:
        np.random.seed(seed)

    token_ids = np.array(token_ids).flatten().astype(np.int64)

    masked_token_ids = token_ids.copy()
    labels = np.full(len(token_ids), -100, dtype=np.int64)
    mask_positions = np.zeros(len(token_ids), dtype=bool)

    num_to_mask = max(1, int(len(token_ids) * mask_prob))
    selected_indices = np.random.choice(len(token_ids), num_to_mask, replace=False)

    for idx in selected_indices:
        labels[idx] = int(token_ids[idx])
        mask_positions[idx] = True

        prob = np.random.rand()
        if prob <= 0.8:
            masked_token_ids[idx] = int(mask_token_id)
        elif prob <= 0.9:
            masked_token_ids[idx] = np.random.randint(3, vocab_size)
        
    return masked_token_ids, labels, mask_positions
    

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token probabilities.
        """
        
        logits = np.dot(hidden_states, self.W) + self.b
        
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs
