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
from transformers import DPRContextEncoderTokenizer, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
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
            'checkpoint/dpr_zalo_v1/query_encoder'
        )
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()

    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            hits = self.searcher.search(question['question'], self.top_k)
            candidate_passages, doc_ids, scores_dpr = [], [], []

            doc_ids = [hit.docid for hit in hits]
            scores_dpr = [hit.score/100 for hit in hits]

            for doc_id, score in zip(doc_ids, scores_dpr):
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()[0]
                res = self.cur.execute("SELECT text FROM documents WHERE id= ?", (str(doc_id), ))
                ctx = res.fetchone()[0]
                passage_vn = (doc_id, wikipage, score, ctx)
                candidate_passages.append(passage_vn)
                        
            question['candidate_passages'] = candidate_passages
        print("Retrieved passages.")
        return data

class HybridRetriever(BaseRetriever):
    def __init__(self, index_path, top_k, db_path) -> None:
        self.ssearcher = LuceneSearcher(index_path)
        self.ssearcher.set_language('vi')
        self.dsearcher = FaissSearcher(
            'checkpoint/indexes/dpr_zalo_v1',
            'checkpoint/dpr_zalo_v1/query_encoder'
        )
        self.hsearcher = HybridSearcher(self.dsearcher, self.ssearcher)
        self.top_k = top_k
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()

    def __call__(self, data):
        print("Retrieving passages...")
        for question in tqdm(data):
            hits = self.hsearcher.search(question['question'], 2000, self.top_k, normalization=True)
            candidate_passages, doc_ids, scores_hybrid = [], [], []

            doc_ids = [hit.docid for hit in hits]
            scores_hybrid = [hit.score/100 for hit in hits]

            for doc_id, score in zip(doc_ids, scores_hybrid):
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()[0]
                res = self.cur.execute("SELECT text FROM documents WHERE id= ?", (str(doc_id), ))
                ctx = res.fetchone()[0]
                passage_vn = (doc_id, wikipage, score, ctx)
                candidate_passages.append(passage_vn)
            print(len(candidate_passages), candidate_passages)
            question['candidate_passages'] = candidate_passages
        print("Retrieved passages.")
        return data

class HybridRetrieverOnline(BaseRetriever):
    def __init__(self, cfg, db_path) -> None:
        self.bm25 = BM25Retriever(cfg.index_path, cfg.top_h, db_path)
        self.dpr = DPROnlineRetriever(cfg.passage_encoder, cfg.query_encoder, cfg.top_k)
    
    def __call__(self, data):
        print("Retrieving passages...")
        data = self.bm25(data)
        for question in tqdm(data):
            contexts = []
            candidate_passages = question['candidate_passages']
            for passage in candidate_passages:
                contexts.append(passage[3])
            candidate_ids = self.dpr(question['question'], contexts)
            candidate_passages = [candidate_passages[i] for i in candidate_ids]
            question['candidate_passages'] = candidate_passages 
        return data

class DPROnlineRetriever:
    def __init__(self, passage_encoder, query_encoder, top_k):
        self.top_k = top_k
        self.context_encoder = DPRContextEncoder.from_pretrained(passage_encoder).cuda()
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_encoder)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(query_encoder).cuda()
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_encoder)
    
    def __call__(self, question, contexts):
        # print(question, contexts)
        question_ids = self.question_tokenizer.encode(question, max_length=256, return_tensors='pt', truncation=True).cuda()
        question_embeddings = self.question_encoder(question_ids).pooler_output.detach().cpu().numpy()
        contexts_ids = self.context_tokenizer(contexts, return_tensors='pt', padding="longest", truncation=True, add_special_tokens=True, max_length=512)
        contexts_embeddings = self.context_encoder(contexts_ids["input_ids"].cuda()).pooler_output.detach().cpu().numpy()
        scores = np.matmul(question_embeddings, contexts_embeddings.T)
        candidate_ids = np.argsort(scores)[0][-self.top_k:][::-1]
        return candidate_ids

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
        self.cur = sqlite3.connect(osp.join(os.getcwd(), db_path)).cursor()
        
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