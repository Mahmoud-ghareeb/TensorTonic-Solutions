import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Pool the [CLS] token representation.
        """
        
        return tanh(hidden_states[:,0,:] @ self.W + self.b)

class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Classify sequences.
        """
        
        pooler = self.pooler.forward(hidden_states)

        return pooler @ self.classifier
        
        
