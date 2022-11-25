from .base import BaseReader
from .bertbasereader import BertReader
from .ensembler import EnsembleReader
from .T5multidoc import T5multidocreader

def build_reader(cfg, tokenizer, db_path):
    if cfg.type == "default":
        return BaseReader(cfg, tokenizer, db_path)
    if cfg.type == "bert_xlm":
        return BertReader(cfg, tokenizer, db_path)
    if cfg.type == "ensemble":
        return EnsembleReader(cfg, tokenizer, db_path)
    if cfg.type == "t5_multidoc":
        return T5multidocreader(cfg, tokenizer, db_path)
    else:
        raise NotImplementedError
