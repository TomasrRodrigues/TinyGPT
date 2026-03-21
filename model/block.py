import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward

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

def __main__():
    # Example usage of TransformerBlock
    d_model = 256
    n_heads = 4
    block_size = 128
    batch_size = 32

    block = TransformerBlock(d_model, n_heads, block_size)
    input_tensor = torch.randn(batch_size, block_size, d_model)

    output = block(input_tensor)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    __main__()