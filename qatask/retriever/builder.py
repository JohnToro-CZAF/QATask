from .tfidf_retriever import TFIDFRetriever

def build_retriever(cfg, tokenizer, db_path):
    if cfg.type == "tf-idf":
        return TFIDFRetriever(cfg, tokenizer, db_path)
    else:
        assert cfg.type == "default", "NotImplemented retriever{}".format(cfg.type)
    