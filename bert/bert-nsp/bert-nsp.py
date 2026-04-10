import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]], 
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Create NSP training examples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    data_set = []
    n = len(documents)

    for idx, sents in enumerate(documents):
        for i in range(len(sents)-1):
            data_set.append((sents[i], sents[i+1], 1))
            r = idx
            while r == idx:
                r = int(np.random.choice(n))

            c = int(np.random.choice(len(documents[r])))

            data_set.append((sents[i], documents[r][c], 0))

    if num_examples and len(data_set) > num_examples:
       data_set = random.sample(data_set, num_examples)
        
    return data_set

class NSPHead:
    """Next Sentence Prediction classification head."""
    
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        Predict IsNext probability.
        """
        
        return softmax(cls_hidden @ self.W + self.b)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
