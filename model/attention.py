import torch
import torch.nn as nn
import math 

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Causal Masking.

    Args:
        d_model (int): Dimension of input embeddings.
        n_heads (int): Number of attention heads.
        block_size (int): Maximum sequence length (for masking).

    Forward Input:
        x (Tensor): Shape (batch_size, seq_len, d_model)

    Returns:
        Tensor: Shape (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask to prevent attention to future positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).bool()
        )

        self.attn_dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Compute multi-head self-attention with causal masking.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project input to query, key, and value tensors
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        mask = self.mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))

        # Softmax over attention scores
        att = torch.softmax(att, dim=-1)

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        out = att @ V

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)
    