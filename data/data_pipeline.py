# data_pipeline.py
from datasets import load_dataset
from tokenizer import Tokenizer
import torch


class TinyGPTDataPipeline:
    """
    Main class for managing the TinyGPT data pipeline.
    Handles loading datasets, tokenization, batching, and sanity checks.
    """
    def __init__(self, block_size=128, batch_size=32, tokenize_method="char"):
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenize_method = tokenize_method

        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.tiny_loader = None

    class TinyDataset(torch.utils.data.Dataset):
        """
        Internal dataset class for batching token sequences for language modeling.
        Each item is a tuple of input and target token tensors.
        """
        def __init__(self, tokens, block_size):
            self.tokens = tokens
            self.block_size = block_size

        def __len__(self):
            return len(self.tokens) - self.block_size

        def __getitem__(self, idx):
            x = self.tokens[idx: idx + self.block_size]
            y = self.tokens[idx + 1: idx + self.block_size + 1]
            return torch.tensor(x), torch.tensor(y)

    @staticmethod
    def prepare_dataloader(dataset, batch_size=32, shuffle=True):
        """
        Wraps a dataset in a PyTorch DataLoader for batching.
        """
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def load_datasets(self, tinystories_subset=50, wikitext_subset=1000):
        """
        Loads TinyStories and WikiText datasets.
        Allows using subsets for quick experiments.
        """
        self.tiny_stories = load_dataset("roneneldan/TinyStories")["train"][:wikitext_subset]
        self.wiki_text = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"][:wikitext_subset]

        self.tiny_subset = load_dataset("roneneldan/TinyStories")["train"][:tinystories_subset]

    def build_tokenizer(self, text):
        """
        Initializes the tokenizer on provided text.
        """
        self.tokenizer = Tokenizer(text, method=self.tokenize_method)

    def preprocess(self, text, tiny_text=None):
        """
        Tokenizes text, splits into train/validation sets, and prepares DataLoaders.
        Optionally prepares a tiny DataLoader for quick tests.
        """
        if self.tokenizer is None:
            self.build_tokenizer(text)

        tokens = self.tokenizer.encode(text)
        n = len(tokens)
        train_tokens = tokens[: int(n * 0.9)]
        val_tokens = tokens[int(n * 0.9):]

        train_dataset = self.TinyDataset(train_tokens, self.block_size)
        val_dataset = self.TinyDataset(val_tokens, self.block_size)

        self.train_loader = self.prepare_dataloader(train_dataset, self.batch_size)
        self.val_loader = self.prepare_dataloader(val_dataset, self.batch_size, shuffle=False)

        self.tiny_loader = None
        if tiny_text is not None:
            tiny_tokens = self.tokenizer.encode(tiny_text)
            tiny_dataset = self.TinyDataset(tiny_tokens, self.block_size)
            self.tiny_loader = self.prepare_dataloader(tiny_dataset, self.batch_size)

        return self.train_loader, self.val_loader, self.tiny_loader, self.tokenizer

    def sanity_check(self, loader=None):
        """
        Checks batch shapes and prints sample decoded data for quick verification.
        """
        loader = loader or self.train_loader
        xb, yb = next(iter(loader))
        assert xb.shape == yb.shape and xb.shape[1] == self.block_size, "Batch shapes mismatch"
        print("Sanity check passed | Batch shapes:", xb.shape, yb.shape)
        x_sample, y_sample = loader.dataset[10]
        print("Decoded sample x:", self.tokenizer.decode(x_sample))
        print("Decoded sample y:", self.tokenizer.decode(y_sample))
        return xb, yb
    

pipeline = TinyGPTDataPipeline(block_size=128, batch_size=32)
pipeline.load_datasets(tinystories_subset=50, wikitext_subset=1000)

pipeline.tiny_stories = load_dataset("roneneldan/TinyStories")["train"][:1000]
pipeline.tiny_subset = load_dataset("roneneldan/TinyStories")["train"][:50]

text = "\n".join(pipeline.tiny_stories["text"])
tiny_text = "\n".join(pipeline.tiny_subset["text"])

train_loader, val_loader, tiny_loader, tokenizer = pipeline.preprocess(text, tiny_text=tiny_text)

xb, yb = pipeline.sanity_check(train_loader)