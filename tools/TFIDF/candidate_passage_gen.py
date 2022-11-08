import sys
import argparse
import logging
sys.path.insert(1, '/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/retriever')
import sqlite3

from TFIDF.tokenizers.vnm_tokenizer import VNMTokenizer
from TFIDF import retriver_drqa
import json

def flatten(l):
    return [item for sublist in l for item in sublist]

def process(cur, ranker, query, k=5):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    doc_wiki = []
    for doc_name in doc_names:
        res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_name), ))
        wikipage = res.fetchone()
        doc_wiki.append(wikipage)
    return doc_wiki 

def main():
  ranker = retriver_drqa.get_class('tfidf')(tfidf_path=args.model)
  con = sqlite3.connect("/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/database/SQLDB/wikisqlite.db")
  cur = con.cursor()

  tokenizer = VNMTokenizer()
  with open("/home/ubuntu/hoang.pn200243/AQ/QATask/sample/e2eqa-train+public_test-v1/test_sample.json") as f:
    file = json.loads(f.read())
  questions = []
  data = file['data']
  for question in data:
    ans = process(cur, ranker, question['question'])
    question['answer'] = ans
  temp = {"data": data}
  with open("/home/ubuntu/hoang.pn200243/AQ/QATask/sample/e2eqa-train+public_test-v1/test_answer2.json", 'w') as f2:
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
  args = parser.parse_args()

  logger.info('Initializing ranker...')
  main()