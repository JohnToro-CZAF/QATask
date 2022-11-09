import os
import sys
from pathlib import PosixPath
DATA_DIR = (
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'qatask/database')
)
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

from .doc_db import DocDB


