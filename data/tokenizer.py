import re

class Tokenizer:

    def __init__(self, text, method="char"):
        self.tokenize = Tokenizer.character_tokenizer if method == "char" else Tokenizer.word_tokenizer
        self.separator = "" if method == "char" else " "
        tokens = self.tokenize(text)
        vocab = sorted(set(tokens))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        return [self.stoi[t] for t in self.tokenize(text)]

    def decode(self, indices):
        return self.separator.join([self.itos[i] for i in indices])

    @staticmethod
    def character_tokenizer(text):
        return list(text)

    @staticmethod
    def word_tokenizer(text):
        return re.findall(r"\w+|[^\w\s]", text)

