import torch
import torch.nn as nn
import math 


def _tensor_preview(tensor, max_items=8):
    """Return a short, readable preview of tensor values."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    preview = flat[:max_items].cpu().tolist()
    return [round(float(v), 4) for v in preview]

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
        Q = self.q_proj(x)  # Query projection
        K = self.k_proj(x)  # Key projection
        V = self.v_proj(x)  # Value projection

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
        att = self.attn_dropout(att)

        # Weighted sum of values
        out = att @ V

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


def run_attention_step_by_step(
    batch_size=2,
    seq_len=4,
    d_model=16,
    n_heads=4,
    block_size=8,
    seed=42,
):
    """
    Run only multi-head self-attention and print each stage.

    Returns:
        Tensor: Output of attention block with shape (batch_size, seq_len, d_model)
    """
    torch.manual_seed(seed)

    attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, block_size=block_size)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    B, T, C = x.shape

    print("Step 0 - Input")
    print(f"x shape: {x.shape}")
    print(f"x sample: {_tensor_preview(x)}")

    with torch.no_grad():
        print("\nStep 1 - Linear projections")
        Q = attn.q_proj(x)
        K = attn.k_proj(x)
        V = attn.v_proj(x)
        print(f"Q shape: {Q.shape}, sample: {_tensor_preview(Q)}")
        print(f"K shape: {K.shape}, sample: {_tensor_preview(K)}")
        print(f"V shape: {V.shape}, sample: {_tensor_preview(V)}")

        print("\nStep 2 - Split into heads")
        Qh = Q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        Kh = K.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        Vh = V.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        print(f"Qh shape: {Qh.shape}")
        print(f"Kh shape: {Kh.shape}")
        print(f"Vh shape: {Vh.shape}")

        print("\nStep 3 - Scaled dot-product scores")
        scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        print(f"scores shape: {scores.shape}")
        print(f"scores sample: {_tensor_preview(scores)}")

        print("\nStep 4 - Causal masking")
        mask = attn.mask[:T, :T]
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        print(f"mask shape: {mask.shape}")
        print(f"mask row 0: {mask[0].int().tolist()}")
        print(f"masked_scores sample: {_tensor_preview(masked_scores[torch.isfinite(masked_scores)])}")

        print("\nStep 5 - Softmax attention weights")
        weights = torch.softmax(masked_scores, dim=-1)
        print(f"weights shape: {weights.shape}")
        print(f"weights sample: {_tensor_preview(weights)}")

        print("\nStep 6 - Weighted value sum")
        head_out = weights @ Vh
        print(f"head_out shape: {head_out.shape}")
        print(f"head_out sample: {_tensor_preview(head_out)}")

        print("\nStep 7 - Merge heads + output projection")
        merged = head_out.transpose(1, 2).contiguous().view(B, T, C)
        out = attn.out_proj(merged)
        print(f"merged shape: {merged.shape}")
        print(f"out shape: {out.shape}")
        print(f"out sample: {_tensor_preview(out)}")

    return out


def __main__():
    run_attention_step_by_step()

if __name__ == "__main__":
    __main__()