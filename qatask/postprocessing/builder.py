from .base import BasePostProcessor
def build_postprocessor(cfg):
    if cfg.type == "default":
        return BasePostProcessor(cfg)
    else:
        return NotImplementedError("NotImplemented postprocessor{}".format(cfg.type))