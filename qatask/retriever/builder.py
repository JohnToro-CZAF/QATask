from .tfidf_retriever import TFIDFRetriever
from .serini_retriever import ColbertRetriever, DPRRetriever, ANCERetriever, BM25Retriever, HybridRetriever, HybridRetrieverOnline, MultilingualDPRBM25Retriever
from .tokenized_retriever import TokBM25Retriever
from .dual_retriever import DualBM25Retriever


def build_retriever(cfg, tokenizer, db_path):
    if cfg.type == "tf-idf":
        return TFIDFRetriever(cfg, tokenizer, db_path)
    elif cfg.type == "colbertv2":
        return ColbertRetriever(cfg.index_path, cfg.top_k, db_path)
    elif cfg.type == "dpr":
        return DPRRetriever(cfg.index_path, cfg.top_k, db_path)
    elif cfg.type == "ance":
        return ANCERetriever(cfg.index_path, cfg.top_k, db_path)
    elif cfg.type == "bm25":
        return BM25Retriever(cfg.index_path, cfg.top_k, db_path)
    elif cfg.type == "hybrid":
        return HybridRetriever(cfg.index_path, cfg.top_k, db_path)
    elif cfg.type == "hybrido":
        return HybridRetrieverOnline(cfg, db_path)
    elif cfg.type == "dual_bm25":
        return DualBM25Retriever(cfg, db_path)
    elif cfg.type == "tok_bm25":
        return TokBM25Retriever(cfg, db_path)
    elif cfg.type == "multilingual_dpr_bm25":
        return MultilingualDPRBM25Retriever(cfg, db_path)
    else:
        assert cfg.type == "default", "NotImplemented retriever{}".format(cfg.type)
    