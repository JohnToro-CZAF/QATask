from qatask.retriever.tfidf.doc_db import DocDB
import os
import sqlite3
import os.path as osp

class BaseReader:
    def __init__(self, cfg, tokenizer, db_path) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
        
    
    def __call__(self, data):
        return data