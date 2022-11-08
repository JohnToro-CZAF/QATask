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
import sys

sys.path.insert(1, '/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/retriever')

from TFIDF import retriver_drqa

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
ranker = retriver_drqa.get_class('tfidf')(tfidf_path=args.model)

import sqlite3
con = sqlite3.connect("/home/ubuntu/hoang.pn200243/AQ/QATask/qatask/database/SQLDB/wikisqlite.db")
cur = con.cursor()

# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=10):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score', 'Wiki Page']
    )
    doc_wiki = []
    for doc_name in doc_names:
        res = cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_name), ))
        wikipage = res.fetchone()
        doc_wiki.append(wikipage)
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], doc_wiki[i]])
    print(table)


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
