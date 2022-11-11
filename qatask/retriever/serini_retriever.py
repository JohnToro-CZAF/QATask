from pyserini.search import FaissSearcher
from .base import BaseRetriever

searcher = FaissSearcher(
    'checkpoint/indexes/colbert-v2/index',
    'castorini/tct_colbert-v2-hnp-msmarco'
)
hits = searcher.search('what is a lobster roll')

for i in range(0, 5):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')

class ColbertRetriever(BaseRetriever):
    def __init__(self, index_path, top_k):
        self.searcher = FaissSearcher(
            index_path,
            'castorini/tct_colbert-v2-hnp-msmarco'
        )
        self.docdb = 1
    
    def __call__(self, data):
        for question in data:
            hits = searcher.search(question['question'])
        pass