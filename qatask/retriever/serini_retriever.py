from pyserini.search import FaissSearcher
from qatask.retriever.tfidf.doc_db import DocDB
from .base import BaseRetriever

class ColbertRetriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'castorini/tct_colbert-v2-hnp-msmarco'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
    
    def __call__(self, data):
        for question in data:
            hits = self.searcher.search(question['question'])
            candidate_passages = []
            for i in range(0, self.top_k):
                doc_id = hits[i].docid
                passage_vn = self.docdb.get_doc_text(doc_id)
                candidate_passages.append(passage_vn)
            question['candidate_passages'] = candidate_passages
        return data

class DPRRetriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'facebook/dpr-question_encoder-multiset-base'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
    
    def __call__(self, data):
        for question in data:
            hits = self.searcher.search(question['question'])
            candidate_passages = []
            for i in range(0, self.top_k):
                doc_id = hits[i].docid
                passage_vn = self.docdb.get_doc_text(doc_id)
                candidate_passages.append(passage_vn)
            question['candidate_passages'] = candidate_passages
        return data

class ANCERetriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path):
        self.searcher = FaissSearcher(
            index_path,
            'castorini/ance-msmarco-passage'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
    
    def __call__(self, data):
        for question in data:
            hits = self.searcher.search(question['question'])
            candidate_passages = []
            for i in range(0, self.top_k):
                doc_id = hits[i].docid
                passage_vn = self.docdb.get_doc_text(doc_id)
                candidate_passages.append(passage_vn)
            question['candidate_passages'] = candidate_passages
        return data

if __name__ == "__main__":
    searcher = FaissSearcher(
    'checkpoint/indexes/colbert-v2/index',
    'castorini/tct_colbert-v2-hnp-msmarco'
    )
    hits = searcher.search('what is a lobster roll')

    for i in range(0, 5):
        print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')