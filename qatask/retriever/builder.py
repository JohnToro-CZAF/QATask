from .tfidf_retriever import TFIDFRetriever
from .serini_retriever import ColbertRetriever, DPRRetriever, ANCERetriever, BM25Retriever, HybridRetriever, HybridRetrieverOnline
from .dual_retriever import DualBM25Retriever
from .sentence_retriever import MiniLMBM25Retriever


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
    elif cfg.type == "minilm_bm25":
        return MiniLMBM25Retriever(cfg, db_path)
    else:
        assert cfg.type == "default", "NotImplemented retriever{}".format(cfg.type)
    