from pyserini.search import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from qatask.retriever.tfidf.doc_db import DocDB
from .base import BaseRetriever
import sqlite3
import os.path as osp
import os
from tqdm import tqdm
import numpy as np
# from CocCocTokenizer import PyTokenizer

class TokBM25Retriever(BaseRetriever):
    def __init__(self, cfg, db_path):
        # self.T = PyTokenizer(load_nontone_data=True)
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')
        self.top_k = cfg.top_k
        self.cur = sqlite3.connect(osp.join(os.getcwd(), db_path)).cursor()
        
    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            query = " ".join(self.T.word_tokenize(question['question'], tokenize_option=0)) 
            # query = question['question']
            hits1 = self.searcher.search(query, self.top_k)
            hits2 = self.searcher.search(question['question'], self.top_k)
            hits = list(set(hits1+hits2))
            candidate_passages, doc_ids, scores_bm25 = [], [], []

            doc_ids = [hit.docid for hit in hits]
            scores_bm25 = [hit.score/100 for hit in hits]

            for idx, (doc_id, score) in enumerate(zip(doc_ids, scores_bm25)):
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()[0]
                res = self.cur.execute("SELECT text FROM documents WHERE id= ?", (str(doc_id), ))
                ctx = res.fetchone()[0]
                tmp_ctx = hits[idx].raw
                passage_vn = (doc_id, wikipage, score, tmp_ctx)
                candidate_passages.append(passage_vn)
            question['candidate_passages'] = candidate_passages
            question['question'] = query
        print("Retrieved passages.")
        return data