import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import torch
import unittest
from data.data_pipeline import TinyGPTDataPipeline


# --- Helpers ---

SAMPLE_TEXT = "hello world this is a simple test for the tinygpt data pipeline " * 50
TINY_TEXT = "tiny story about a cat " * 20


# --- TinyDataset Tests ---

class TestTinyDataset(unittest.TestCase):
    """Tests for the internal TinyDataset class."""

    def test_length_is_correct(self):
        """Dataset length should be total tokens minus block size."""
        tokens = list(range(200))
        block_size = 32
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, block_size)
        self.assertEqual(len(dataset), 200 - 32)

    def test_item_shapes_are_correct(self):
        """Each item should return x and y tensors of length block_size."""
        tokens = list(range(200))
        block_size = 32
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, block_size)
        x, y = dataset[0]
        self.assertEqual(x.shape, torch.Size([block_size]))
        self.assertEqual(y.shape, torch.Size([block_size]))

    def test_y_is_x_shifted_by_one(self):
        """y should be x shifted one position to the right (next token prediction)."""
        tokens = list(range(200))
        block_size = 32
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, block_size)
        x, y = dataset[0]
        self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_returns_tensors(self):
        """Items should be PyTorch tensors."""
        tokens = list(range(100))
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, 16)
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)


# --- prepare_dataloader Tests ---

class TestPrepareDataloader(unittest.TestCase):
    """Tests for the static prepare_dataloader method."""

    def test_returns_dataloader(self):
        """Should return a PyTorch DataLoader."""
        tokens = list(range(200))
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, 32)
        loader = TinyGPTDataPipeline.prepare_dataloader(dataset, batch_size=8)
        self.assertIsInstance(loader, torch.utils.data.DataLoader)

    def test_batch_size_is_respected(self):
        """Batches should match the requested batch size."""
        tokens = list(range(500))
        dataset = TinyGPTDataPipeline.TinyDataset(tokens, 32)
        loader = TinyGPTDataPipeline.prepare_dataloader(dataset, batch_size=16)
        xb, yb = next(iter(loader))
        self.assertEqual(xb.shape[0], 16)


# --- build_tokenizer Tests ---

class TestBuildTokenizer(unittest.TestCase):
    """Tests for tokenizer initialization."""

    def test_tokenizer_is_set_after_build(self):
        """Tokenizer should not be None after build_tokenizer is called."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        pipeline.build_tokenizer(SAMPLE_TEXT)
        self.assertIsNotNone(pipeline.tokenizer)

    def test_tokenizer_can_encode_and_decode(self):
        """Tokenizer should be able to encode text and decode it back."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        pipeline.build_tokenizer(SAMPLE_TEXT)
        tokens = pipeline.tokenizer.encode("hello")
        decoded = pipeline.tokenizer.decode(tokens)
        self.assertIn("hello", decoded)


# --- preprocess Tests ---

class TestPreprocess(unittest.TestCase):
    """Tests for the preprocess method."""

    def test_returns_loaders_and_tokenizer(self):
        """preprocess should return train, val, tiny loaders and a tokenizer."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        # Build tokenizer on combined text so all characters in TINY_TEXT are in the vocabulary
        pipeline.build_tokenizer(SAMPLE_TEXT + TINY_TEXT)
        train_loader, val_loader, tiny_loader, tokenizer = pipeline.preprocess(SAMPLE_TEXT, tiny_text=TINY_TEXT)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(tiny_loader)
        self.assertIsNotNone(tokenizer)

    def test_tiny_loader_is_none_when_not_provided(self):
        """tiny_loader should be None if no tiny_text is passed."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        _, _, tiny_loader, _ = pipeline.preprocess(SAMPLE_TEXT)
        self.assertIsNone(tiny_loader)

    def test_train_val_split_is_90_10(self):
        """Train set should be roughly 90% and val 10% of total tokens."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        pipeline.build_tokenizer(SAMPLE_TEXT)
        tokens = pipeline.tokenizer.encode(SAMPLE_TEXT)
        n = len(tokens)
        pipeline.preprocess(SAMPLE_TEXT)
        train_len = len(pipeline.train_loader.dataset)
        val_len = len(pipeline.val_loader.dataset)
        self.assertEqual(train_len, int(n * 0.9) - 32)
        self.assertEqual(val_len, (n - int(n * 0.9)) - 32)

    def test_batch_shapes_are_correct(self):
        """Batches from train loader should match block_size."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        train_loader, _, _, _ = pipeline.preprocess(SAMPLE_TEXT)
        xb, yb = next(iter(train_loader))
        self.assertEqual(xb.shape[1], 32)
        self.assertEqual(yb.shape[1], 32)


# --- sanity_check Tests ---

class TestSanityCheck(unittest.TestCase):
    """Tests for the sanity_check method."""

    def test_sanity_check_passes(self):
        """sanity_check should not raise any assertion errors on valid data."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        pipeline.preprocess(SAMPLE_TEXT)
        try:
            pipeline.sanity_check()
        except AssertionError:
            self.fail("sanity_check raised AssertionError unexpectedly.")

    def test_sanity_check_returns_batches(self):
        """sanity_check should return xb and yb tensors."""
        pipeline = TinyGPTDataPipeline(block_size=32, batch_size=8)
        pipeline.preprocess(SAMPLE_TEXT)
        xb, yb = pipeline.sanity_check()
        self.assertIsInstance(xb, torch.Tensor)
        self.assertIsInstance(yb, torch.Tensor)


if __name__ == "__main__":
    unittest.main()