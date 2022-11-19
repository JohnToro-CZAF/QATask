from .base import BaseReader
from .bertbasereader import BertReader
from .ensembler import EnsembleReader

def build_reader(cfg, tokenizer, db_path):
    if cfg.type == "default":
        return BaseReader(cfg, tokenizer, db_path)
    if cfg.type == "bert_xlm":
        return BertReader(cfg, tokenizer, db_path)
    if cfg.type == "ensemble":
        return EnsembleReader(cfg, tokenizer, db_path)
    else:
        raise NotImplementedError
