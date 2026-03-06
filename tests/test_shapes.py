import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.tinygpt import TinyGPT


model = TinyGPT(
    vocab_size=100,
    block_size=128,
    d_model=256,
    n_heads=4,
    n_layers=4
)

x = torch.randint(0,100,(32,128))

logits = model(x)

print(logits.shape)