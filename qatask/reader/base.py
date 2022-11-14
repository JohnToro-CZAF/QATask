from qatask.retriever.tfidf.doc_db import DocDB
import os
import sqlite3
import os.path as osp
import json
import time

class BaseReader:
    def __init__(self, cfg, tokenizer, db_path) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    
    def logging(self, data) -> None:
        with open(self.cfg.logpth + "_" + str(time.time()), 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    def __call__(self, data):
        return data