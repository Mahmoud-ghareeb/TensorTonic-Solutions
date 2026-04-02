import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: Conv -> ReLU -> Conv -> Add Skip -> ReLU
        """
        if self.downsample:
            skip = x @ self.W_proj
        else:
            skip = x
            
        conv = x @ self.W1
        rlu = relu(conv)
        conv = rlu @ self.W2

        conv += skip

        return relu(conv)
        

class ResNet18:
    """
    Simplified ResNet-18 architecture.
    
    Structure:
    - conv1: 3 -> 64 channels
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels (first block downsamples)
    - layer3: 2 BasicBlocks, 256 channels (first block downsamples)
    - layer4: 2 BasicBlocks, 512 channels (first block downsamples)
    - fc: 512 -> num_classes
    """
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # Build layers - YOUR CODE HERE
        self.layer1 = [BasicBlock(64, 64), BasicBlock(64, 64)]  # 2 blocks: 64 -> 64
        self.layer2 = [BasicBlock(64, 128, downsample=True), BasicBlock(128, 128)]   # 2 blocks: 64 -> 128 (first downsamples)
        self.layer3 = [BasicBlock(128, 256, downsample=True), BasicBlock(256, 256)]  # 2 blocks: 128 -> 256 (first downsamples)
        self.layer4 = [BasicBlock(256, 512, downsample=True), BasicBlock(512, 512)]  # 2 blocks: 256 -> 512 (first downsamples)
        
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through ResNet-18.
        """
        x = x @ self.conv1
        for layer in self.layer1:
            x = layer.forward(x)
        for layer in self.layer2:
            x = layer.forward(x)
        for layer in self.layer3:
            x = layer.forward(x)
        for layer in self.layer4:
            x = layer.forward(x)

        return x @ self.fc
