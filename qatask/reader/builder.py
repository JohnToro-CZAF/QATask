from .base import BaseReader
from .bertbase import BertBase

def build_reader(cfg, tokenizer):
    if cfg.type == "default":
        return BaseReader(cfg, tokenizer)
    if cfg.type == "bert_xlm":
        return BertBase(cfg, tokenizer)
    else:
        raise NotImplementedError
