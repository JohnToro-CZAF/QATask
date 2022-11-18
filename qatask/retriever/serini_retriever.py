from pyserini.search import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from qatask.retriever.tfidf.doc_db import DocDB
from .base import BaseRetriever
import sqlite3
import os.path as osp
import os
from tqdm import tqdm
import ipdb
from underthesea import word_tokenize

class ColbertRetriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'castorini/tct_colbert-v2-hnp-msmarco'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
        self.setup_translator()

    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            hits = self.searcher.search(word_tokenize(question['question']))
            candidate_passages = []
            for i in range(0, self.top_k):
                doc_id = hits[i].docid
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()
                passage_vn = (doc_id, wikipage)
                candidate_passages.append(passage_vn)
            question['candidate_passages'] = candidate_passages
        return data
    
    def setup_translator(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation").cuda()
    
    def translate(self, question):
        outputs = self.model.generate(self.tokenizer(question, return_tensors="pt", padding=True).input_ids.cuda(), max_length=512)
        en_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return en_text[0][4:]

class DPRRetriever(ColbertRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'facebook/dpr-question_encoder-multiset-base'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            hits = self.searcher.search(question['question'], self.top_k)
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
            question['candidate_passages'] = candidate_passages
        print("Retrieved passages.")
        return data

class HyrbidRetriver(BaseRetriever):
    def __init__(self, cfg) -> None:
        self.sieve = BM25Retriever(cfg.sieve)
        # self.dryer = 
        pass

class ANCERetriever(ColbertRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'castorini/ance-msmarco-passage'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    
class BM25Retriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_language('vn')
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    
    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            hits = self.searcher.search(question['question'], self.top_k)
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
            question['candidate_passages'] = candidate_passages
        print("Retrieved passages.")
        return data

if __name__ == "__main__":
    searcher = FaissSearcher(
    'checkpoint/indexes/colbert-v2',
    'castorini/tct_colbert-v2-hnp-msmarco'
    )
    hits = searcher.search('what is a lobster roll')

    for i in range(0, 5):
        print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')