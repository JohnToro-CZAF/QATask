from .base import BaseRetriever
from .tfidf.tfidf_doc_ranker import TfidfDocRanker
from .tfidf.build_tfidf import *
import sqlite3
import os.path as osp
import os
import math

def process(cur, ranker, query, k=5):
    doc_ids, doc_scores = ranker.closest_docs(query, k)
    doc_wiki = []
    for doc_id in doc_ids:
        res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
        wikipage = res.fetchone()
        doc_wiki.append(wikipage)
    return doc_wiki 

class TFIDFRetriever(BaseRetriever):
    def __init__(self, cfg, tokenizer, db_path) -> None:
        super().__init__()
        self.cfg = cfg
        self.top_k = cfg.top_k
        filename = self.building_tfidf(tokenizer)
        self.tfidf_ranker = TfidfDocRanker(tfidf_path = filename, tokenizer=tokenizer)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.top_kcur = con.cursor()
    
    def building_tfidf(self, tokenizer):
        print("Counting words")
        count_matrix, doc_dict = get_count_matrix(self.cfg, 'sqlite', {'db_path': self.cfg.db_path}, tokenizer)
        tfidf = get_tfidf_matrix(count_matrix)
        freqs = get_doc_freqs(count_matrix)
        basename = osp.splitext(osp.basename(self.cfg.db_path))[0]
        basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
        basename += str(time.time())
        filename = os.path.join(self.cfg.checkpoint, basename)
        print("Saving to %s" % filename)
        ##Quickfix for tokenizer name
        metadata = {
        'doc_freqs': freqs,
        'tokenizer': 'vnm',
        'hash_size': int(math.pow(2, self.cfg.hash_size)),
        'ngram': args.ngram,
        'doc_dict': doc_dict,
        }
        data = {
            'data': tfidf.data,
            'indices': tfidf.indices,
            'indptr': tfidf.indptr,
            'shape': tfidf.shape,
            'metadata': metadata
        }
        np.savez(filename, **data)
        print("TFIDF model saved...")
        print('Done')
        return filename 

    def __call__(self, data):
        for question in data:
            ans = process(self.cur, self.tfidf_ranker, question['question'], self.top_k)
            question['answer'] = ans
        return data
