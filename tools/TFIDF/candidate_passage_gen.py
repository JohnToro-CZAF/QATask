import os, sys
import argparse
import logging
import sqlite3

from drqa import retriever
from drqa import tokenizers
# from TFIDF.tokenizers.vnm_tokenizer import VNMTokenizer
# from TFIDF import retriver_drqa
import json

def flatten(l):
    return [item for sublist in l for item in sublist]

def process(cur, ranker, query, k=5):
    doc_ids, doc_scores = ranker.closest_docs(query, k)
    doc_wiki = []
    for doc_id in doc_ids:
        res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
        wikipage = res.fetchone()
        doc_wiki.append(wikipage)
    return doc_wiki 

def main(args):
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    con = sqlite3.connect(os.path.join(os.getcwd(), "qatask/database/wikipedia_db/wikisqlite.db"))
    cur = con.cursor()

    tokenizer = tokenizers.get_class(args.tokenizer)
    with open(os.path.join(os.getcwd(), "qatask/database/datasets/test_sample.json")) as f:
        file = json.loads(f.read())

    questions = []
    data = file['data']
    for question in data:
        ans = process(cur, ranker, question['question'])
        question['answer'] = ans
    temp = {"data": data}
    
    with open(os.path.join(os.getcwd(), "qatask/database/datasets/test_answer_submission.json"), 'w') as f2:
        json.dump(data, f2, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='simple')
    args = parser.parse_args()

    logger.info('Initializing ranker...')

    main(args)