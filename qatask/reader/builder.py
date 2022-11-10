from .base import BaseReader

def build_reader(cfg, tokenizer):
    if cfg.type == "default":
        return BaseReader(cfg, tokenizer)
    else:
        raise NotImplementedError
