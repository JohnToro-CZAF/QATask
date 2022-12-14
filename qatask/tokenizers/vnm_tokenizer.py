# -*- coding: utf-8 -*-
from underthesea import word_tokenize, sent_tokenize, text_normalize
from .tokenizer import Tokens, Tokenizer
import time

class VNMTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        self.annotations = set()
        pass
    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    def tokenize(self, text):
        text = text_normalize(text)
        cleaned_sentences = word_tokenize(text)
        return Tokens(cleaned_sentences, self.annotations)