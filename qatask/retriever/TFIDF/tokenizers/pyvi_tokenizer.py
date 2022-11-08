# -*- coding: utf-8 -*-
from underthesea import word_tokenize, text_normalize, sent_tokenize
from pyvi import ViTokenizer
from .tokenizer import Tokens, Tokenizer

class PIVITokenizer(Tokenizer):
  def __init__(self, **kwargs):
    self.annotations = set()
    pass
  def flatten(l):
      return [item for sublist in l for item in sublist]
  def tokenize(self, text):
    sentences = sent_tokenize(text)
    cleaned_sentences = [text_normalize(sent) for sent in sentences]
    cleaned_sentences = [ViTokenizer.tokenize(sent) for sent in cleaned_sentences]
    data = [sentence.split(" ") for sentence in cleaned_sentences]
    return Tokens(data, self.annotations)