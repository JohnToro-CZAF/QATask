import os
from .. import DATA_DIR
import sys
DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia_db/wikisqlite.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
    'elastic_url': 'localhost:9200'
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker