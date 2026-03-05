import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of multi-head self-attention and a feedforward network.
    Each sub-layer is followed by layer normalization and a residual connection.
    """
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttention(d_model, n_heads, block_size)
        self.ff = FeedForward(d_model, d_model * 4)


    def forward(self, x):
        
        x = x + self.attn(self.ln1(x))  # Self-attention with residual connection
        x = x + self.ff(self.ln2(x))    # Feedforward with residual connection

        return x