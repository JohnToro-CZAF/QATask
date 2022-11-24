from .bm25 import BM25PostProcessor

class EnsemblePostProcessor(BM25PostProcessor):
    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def postprocess(self, query, top_docs, top_docs_scores):
        pass