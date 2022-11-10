import torch
import torch.nn as nn

from .base import BaseReader

class BertBase(BaseReader):
  # class __name__
  def __init__(self, cfg, tokenizer) -> None:
    super().__init__(cfg, tokenizer)
  
  def __call__(self, data):
    
    return data