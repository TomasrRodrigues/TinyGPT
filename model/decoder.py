import torch
import torch.nn as nn
from .block import TransformerBlock


class Decoder(nn.Module):
    def __init__(self, vocab_size, n_layers, d_model, n_heads, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, device=idx.device) 

        pos_emb = self.pos_embedding(pos)

        # tok_emb is (B, T, d_model), pos_emb is (T, d_model)
        # PyTorch "broadcasts" the positions across the batch
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits