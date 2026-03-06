import torch
import torch.nn as nn
from .block import TransformerBlock


class Decoder(nn.Module):
    """
    Transformer decoder stack.

    This module applies a sequence of Transformer blocks to the
    input hidden representations. It does not contain embeddings
    or the language modeling head — those are handled by TinyGPT.
    """

    def __init__(self, n_layers, d_model, n_heads, block_size):
        """
        Initialize the decoder stack.

        Args:
            n_layers (int): Number of transformer blocks.
            d_model (int): Embedding / hidden dimension.
            n_heads (int): Number of attention heads.
            block_size (int): Maximum sequence length.
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])


    def forward(self, x):
        """
        Forward pass through the transformer stack.

        Args:
            x (Tensor): Input hidden states of shape (B, T, d_model)

        Returns:
            Tensor: Output hidden states of shape (B, T, d_model)
        """
        
        for layer in self.layers:
            x = layer(x)


        return x