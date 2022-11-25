from .base import BasePostProcessor
from .bm25 import BM25PostProcessor
# from .vireader import ViPostProcessor
def build_postprocessor(cfg, db_path):
    if cfg.type == "default":
        return BasePostProcessor(cfg, db_path)
    elif cfg.type == "bm25":
        return BM25PostProcessor(cfg, db_path)
    # elif cfg.type == "vi":
    #     return ViPostProcessor(cfg, db_path)
    else:
        return NotImplementedError("NotImplemented postprocessor{}".format(cfg.type))