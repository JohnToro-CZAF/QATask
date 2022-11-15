import json
import os
import re
import sys
import string

from tokenizers import BertWordPieceTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering

import torch
import torch.utils.data import DataLoader, RandomSampler, SequentialSampler

gpu = torch.device('cuda')
# ======================= Downloading data =====================
max_seq_length = 384
batch_size = 16
epochs = 4

# Tokenizer initialization
slow_tokenizer = XLMRobertaTokenizer.from_pretrained('bert-base-uncased')
if not os.path.exists('bert_base_uncased/'):
  os.makedirs("bert_base_uncased/")
slow_tokenizer.save_pretrained('bert_base_uncased/')
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

# Preparing dataset
class Sample:
  def __init__(self, question, context, start_char_idx=None, answer_text=None, 
  all_answers=None) -> None:
    self.question = question
    self.context = context
    self.start_char_idx = start_char_idx
    self.answer_text = answer_text
    self.all_answers = all_answers
    self.skip = False
    self.start_token_idx = -1
    self.end_token_idx = -1
  def preprocess(self):
    context = " ".join(str(self.context).split())
    question = " ".join(str(self.question).split())
    tokenized_context = tokenizer.encode(context)
    tokenized_question = tokenizer.encode(question)
