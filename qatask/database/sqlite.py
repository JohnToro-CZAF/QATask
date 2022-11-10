"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import sys
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
                # doc is a list of dicts
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((doc['id'], doc['text'], doc['wikipage']))
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    print(save_path)
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, wikipage);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for triple in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(triple)
            c.executemany("INSERT INTO documents VALUES (?,?,?)", triple)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()

class SQLiteDatabase():
    def __init__(self, cfg):
        self.data_path = cfg.dataset_path
        self.save_path = cfg.database_path
        store_contents(self.data_path, self.save_path, cfg.preprocess, cfg.num_workers)

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='../../sample/data_wiki_cleaned')
    parser.add_argument('save_path', type=str, help='../../SQLDB/wikisqlite.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    print(args.save_path)
    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers
    )