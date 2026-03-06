
import torch
import torch.nn as nn

from .embeddings import TokenEmbedding, PositionalEmbedding
from .block import TransformerBlock
from .decoder import Decoder

class TinyGPT(nn.Module):

    def __init__(self, vocab_size, block_size, d_model=256, n_heads=4, n_layers=4):
        super().__init__()

        # embeddings
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(block_size, d_model)

        # transformer decoder        
        self.decoder = Decoder(n_layers, d_model, n_heads, block_size)

        # final norm
        self.ln = nn.LayerNorm(d_model)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size)

        # weight tying
        self.lm_head.weight = self.token_emb.embedding.weight   

    def forward(self, x):
        tok = self.token_emb(x)
        pos = self.pos_emb(x)

        x = tok + pos

        x = self.decoder(x)

        x = self.ln(x)

        logits = self.lm_head(x)

        return logits