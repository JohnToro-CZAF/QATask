from qatask.retriever.tfidf.doc_db import DocDB

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import argparse
import sqlite3
import json
import os
import sys
import logging
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample: {"id": "doc1", "contents": "title1\ncontents of doc one."}
def store_contents(doc_db, save_path, model, tokenizer):
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    doc_ids = doc_db.get_doc_ids()
    with open(save_path, "w") as fp:
        for doc_id in tqdm(doc_ids):
            vn_text = "vi: " + doc_db.get_doc_text(doc_id)
            outputs = model.generate(tokenizer(vn_text, return_tensors="pt", padding=True).input_ids.to(device), max_length=512)
            en_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][4:]
            temp = {
                "id": str(doc_id),
                "contents": en_text + "\n"
            }
            json.dump(temp, fp)
            fp.write("\n")
    
class FaissDatabase():
    def __init__(self, cfg):
        self.doc_db = DocDB(cfg.db_path)
        self.tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")  
        self.model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation")

        store_contents(self.doc_db, cfg.save_path, self.model, self.tokenizer)
        self.doc_db.__exit__()

if __name__ == '__main__':
    # debugging purpose
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', default="qatask/database/wikipedia_db/wikisqlite.db", type=str)
    parser.add_argument('--save-path', default="qatask/database/wikipedia_faiss/wikipedia_pyserini_format.jsonl", type=str)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation")
    tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")  
    doc_db = DocDB(args.db_path)
    store_contents(doc_db, args.save_path, model.cuda(), tokenizer)
    doc_db.__exit__()