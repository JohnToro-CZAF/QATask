from pyserini.search import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
from qatask.retriever.tfidf.doc_db import DocDB
from .base import BaseRetriever
import sqlite3
import os.path as osp
import os
from tqdm import tqdm
# from CocCocTokenizer import PyTokenizer

class DualBM25Retriever(BaseRetriever):
    def __init__(self, cfg, db_path):
        # self.T = PyTokenizer(load_nontone_data=True)
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')

        self.title_searcher = LuceneSearcher(cfg.index_path_title)
        self.title_searcher.set_language('vn')

        self.top_passage = cfg.top_passage
        self.top_title = cfg.top_title
        self.cur = sqlite3.connect(osp.join(os.getcwd(), db_path)).cursor()
        self.title_cur = sqlite3.connect(osp.join(os.getcwd(), cfg.db_path_title)).cursor()

    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            query = question['question']
            hits = self.searcher.search(query, self.top_passage)
            candidate_passages, doc_ids, scores_bm25 = [], [], []

            doc_ids = [hit.docid for hit in hits]
            scores_bm25 = [hit.score/100 for hit in hits]

            for doc_id, score in zip(doc_ids, scores_bm25):
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()[0]
                res = self.cur.execute("SELECT text FROM documents WHERE id= ?", (str(doc_id), ))
                ctx = res.fetchone()[0]
                passage_vn = (doc_id, wikipage, score, ctx)
                candidate_passages.append(passage_vn)
            
            hits_title = self.title_searcher.search(question['question'], self.top_title)
            for hit in hits_title:
                res = self.title_cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(hit.docid), ))
                wikipage = res.fetchone()[0]
                res = self.cur.execute("SELECT text FROM documents WHERE wikipage = ?", (str(wikipage), ))
                ctx = res.fetchone()[0]
                passage_vn = (hit.docid, wikipage, hit.score/100, ctx)
                candidate_passages.append(passage_vn)

            question['candidate_passages'] = candidate_passages
        print("Retrieved passages.")
        return data