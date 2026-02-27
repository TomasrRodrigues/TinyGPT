import re

class Tokenizer:
    """Builds a vocabulary from text and converts between strings and integer indices."""

    def __init__(self, text, method="char"):
        """Build vocabulary from the given text. Method is "char" or "word"."""
        self.tokenize = Tokenizer.character_tokenizer if method == "char" else Tokenizer.word_tokenizer
        self.separator = "" if method == "char" else " "
        tokens = self.tokenize(text)
        vocab = sorted(set(tokens))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        """String to list of token indices."""
        return [self.stoi[t] for t in self.tokenize(text)]

    def decode(self, indices):
        """List of token indices back to string."""
        return self.separator.join([self.itos[i] for i in indices])

    @staticmethod
    def character_tokenizer(text):
        """Split text into individual characters."""
        return list(text)

    @staticmethod
    def word_tokenizer(text):
        """Split text into words and punctuation."""
        return re.findall(r"\w+|[^\w\s]", text)
        return re.findall(r"\w+|[^\w\s]", text)

