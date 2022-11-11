from .tfidf_retriever import TFIDFRetriever
from .serini_retriever import ColbertRetriever

def build_retriever(cfg, tokenizer, db_path):
    if cfg.type == "tf-idf":
        return TFIDFRetriever(cfg, tokenizer, db_path)
    elif cfg.type == "colbertv2":
        return ColbertRetriever(cfg.index_path, cfg.top_k)
    else:
        assert cfg.type == "default", "NotImplemented retriever{}".format(cfg.type)
    