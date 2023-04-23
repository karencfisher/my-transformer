from tensorflow.keras.utils import pad_sequences
import numpy as np


class WordTokenizer:
    '''
    Inelegant word tokenizer

    Fit method to get vocabularly
    Transform method to tokenize corpus
    If specified, will also produce and attention mask, and return
    two arrays, the token ids and the mask.
    '''
    def __init__(self, make_mask=False):
        self.vocab = None
        self.vocab_reverse = None
        self.make_mask = make_mask

    def fit(self, corpus):
        self.vocab = {'<pad>': 0, '<unk>': 1}
        index = 2
        for s in corpus:
            words = s.strip().split()
            for word in words:
                i = self.vocab.get(word)
                if i is None:
                    self.vocab[word] = index
                    index += 1
        self.vocab_reverse = {value: key for key, value in 
                              self.vocab.items()}
        return len(self.vocab)

    def tokenize(self, document, max_len=100):
        tokens = [self.vocab[word] for word in document.strip().split()]
        tokens_n = len(tokens)
        tokens = [0] * (max_len - tokens_n) + tokens
        if self.make_mask:
            mask = list(map(lambda x: 0 if x == 0 else 1, tokens))
            return [tokens, mask]
        return tokens
    
    def transform(self, corpus, max_len=100):
        assert self.vocab is not None, 'Run fit method on corpus first.'
        tokens = list(map(self.tokenize, corpus))
        return np.array(tokens)

