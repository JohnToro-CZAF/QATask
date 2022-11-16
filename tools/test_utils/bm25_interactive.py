#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
import sys, os
import os.path as osp
import sqlite3
from pyserini.search.lucene import LuceneSearcher
from qatask.retriever.serini_retriever import BM25Retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--index-path', type=str, default="checkpoint/indexes/BM25")
parser.add_argument('--db-path', type=str, default="qatask/database/wikipedia_db/wikisqlite_final1.db")
args = parser.parse_args()

logger.info('Initializing retriever...')

ranker = LuceneSearcher(args.index_path)
ranker.set_language('vn')
con = sqlite3.connect(osp.join(os.getcwd(), args.db_path))
cur = con.cursor()

# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=5):
    hits = ranker.search(query)
    
    # Finding top k passages id and their respective scores
    wiki_pages, doc_ids, scores_bm25 = [], [], []
    i, j = 0, 0
    while (j<k and i<len(hits)):
        doc_id = hits[i].docid
        _res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
        _wikipage = _res.fetchone()
        
        if _wikipage is None:
            i+=1
            continue
        else:
            wiki_pages.append(_wikipage[0])
            doc_ids.append(doc_id)
            scores_bm25.append(hits[i].score/100)
            j += 1
            i += 1 

    candidate_passages = []
    for doc_id, score in zip(doc_ids, scores_bm25):
        res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
        wikipage = res.fetchone()
        if wikipage is not None:
            text_passage = cur.execute("SELECT text FROM documents WHERE id= ?", (str(doc_id), )).fetchone()[0]
            passage_vn = (doc_id, wikipage, score, text_passage)
            candidate_passages.append(passage_vn)

    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Wiki Page', 'Doc Score', 'Content']
    )
    for idx, passage in enumerate(candidate_passages):
        table.add_row([idx + 1, passage[0], passage[1], '%.5g' % passage[2], passage[3]])
    print(table)


banner = """
Interactive BM25 pysirini Retriever
>> process(question, k=1)
>> usage()
"""

def usage():
    print(banner)


code.interact(banner=banner, local=locals())