import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch

from model.block import TransformerBlock

class TestBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.n_heads = 4
        self.block_size = 16
        self.batch_size = 8

    def _build_block(self):
        return TransformerBlock(self.d_model, self.n_heads, self.block_size)

    def test_block_forward_shape_preserved(self):
        block = self._build_block()
        x = torch.randn(self.batch_size, self.block_size, self.d_model)

        out = block(x)

        self.assertEqual(out.shape, (self.batch_size, self.block_size, self.d_model))

    def test_block_forward_dtype_preserved(self):
        block = self._build_block()
        x = torch.randn(self.batch_size, self.block_size, self.d_model, dtype=torch.float32)

        out = block(x)

        self.assertEqual(out.dtype, x.dtype)

    def test_block_forward_is_finite(self):
        block = self._build_block()
        x = torch.randn(self.batch_size, self.block_size, self.d_model)

        out = block(x)

        self.assertTrue(torch.isfinite(out).all().item())

    def test_block_is_deterministic_in_eval_mode(self):
        torch.manual_seed(123)
        block = self._build_block()
        block.eval()
        x = torch.randn(self.batch_size, self.block_size, self.d_model)

        with torch.no_grad():
            out1 = block(x)
            out2 = block(x)

        self.assertTrue(torch.allclose(out1, out2))

    def test_block_backward_pass_produces_gradients(self):
        block = self._build_block()
        x = torch.randn(self.batch_size, self.block_size, self.d_model, requires_grad=True)

        out = block(x)
        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(torch.isfinite(x.grad).all().item())

if __name__ == "__main__":
    unittest.main()