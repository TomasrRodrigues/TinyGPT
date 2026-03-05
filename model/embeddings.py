import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):


    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the token embedding lookup table.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x):
        """
        Look up token embeddings for input indices.
        """
        return self.embedding(x)
    
class PositionalEmbedding(nn.Module):

    def __init__(self, block_size, d_model):
        """
        Initialize the positional embedding lookup table.
        """
        super().__init__()
        self.embedding = nn.Embedding(block_size, d_model)
    
    def forward(self, x):
        """
        Generate positional embeddings for a batch of token sequences.
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        pos= self.embedding(positions)
        return pos.unsqueeze(0).expand(B, T, -1)
    

printable = PositionalEmbedding(100, 32)
print(printable.forward(torch.tensor([[1, 2, 3], [4, 5, 6]])))