import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model.tinygpt import TinyGPT
import unittest

class TestTinyGPTShapes(unittest.TestCase):
    def test_forward_pass_shapes(self):
        vocab_size = 100
        block_size = 128
        d_model = 256
        n_heads = 4
        n_layers = 4

        model = TinyGPT(vocab_size, block_size, d_model, n_heads, n_layers)

        batch_size = 32
        input_tokens = torch.randint(0, vocab_size, (batch_size, block_size))

        logits = model(input_tokens)

        self.assertEqual(logits.shape, (batch_size, block_size, vocab_size))

    def test_forward_pass_shapes2(self):
        vocab_size = 200
        block_size = 128
        d_model = 256
        n_heads = 4
        n_layers = 4

        model = TinyGPT(vocab_size, block_size, d_model, n_heads, n_layers)

        batch_size = 32
        input_tokens = torch.randint(0, vocab_size, (batch_size, block_size))

        logits = model(input_tokens)

        self.assertEqual(logits.shape, (batch_size, block_size, vocab_size))
    

if __name__ == "__main__":
    unittest.main()
