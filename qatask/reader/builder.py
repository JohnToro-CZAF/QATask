from .base import BaseReader
from .bertbase import BertBase

def build_reader(cfg, tokenizer, db_path):
    if cfg.type == "default":
        return BaseReader(cfg, tokenizer)
    if cfg.type == "bert_xlm":
        return BertBase(cfg, tokenizer, db_path)
    else:
        raise NotImplementedError
