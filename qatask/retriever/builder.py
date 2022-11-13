from .tfidf_retriever import TFIDFRetriever
# from .serini_retriever import ColbertRetriever, DPRRetriever, ANCERetriever

def build_retriever(cfg, tokenizer, db_path):
    if cfg.type == "tf-idf":
        return TFIDFRetriever(cfg, tokenizer, db_path)
    # elif cfg.type == "colbertv2":
    #     return ColbertRetriever(cfg.index_path, cfg.top_k, db_path)
    # elif cfg.type == "dpr":
    #     return DPRRetriever(cfg.index_path, cfg.top_k, db_path)
    # elif cfg.type == "ance":
    #     return ANCERetriever(cfg.index_path, cfg.top_k, db_path)
    # else:
    #     assert cfg.type == "default", "NotImplemented retriever{}".format(cfg.type)
    