import unittest
from data.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def test_character_tokenizer(self):
        text = "hello!"
        expected = ['h', 'e', 'l', 'l', 'o', '!']
        result = Tokenizer.character_tokenizer(text)
        self.assertEqual(result, expected)

    def test_word_tokenizer_simple(self):
        text = "Hello, world!"
        expected = ['Hello', ',', 'world', '!']
        result = Tokenizer.word_tokenizer(text)
        self.assertEqual(result, expected)

    def test_word_tokenizer_with_numbers(self):
        text = "abc 123!"
        expected = ['abc', '123', '!']
        result = Tokenizer.word_tokenizer(text)
        self.assertEqual(result, expected)

    def test_word_tokenizer_punctuation(self):
        text = "a.b?c!"
        expected = ['a', '.', 'b', '?', 'c', '!']
        result = Tokenizer.word_tokenizer(text)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
