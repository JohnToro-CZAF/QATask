""" Convert data to MRC format """

import json
import argparse
from tqdm import tqdm
import re
import sqlite3
import os
import os.path as osp
from underthesea import word_tokenize, text_normalize

from qatask.retriever.tfidf.doc_db import DocDB
from qatask.retriever.serini_retriever import BM25Retriever


class _BM25Retriever(BM25Retriever):
    def __init__(self, cfg):
        super().__init__(index_path=cfg.index_path, top_k=cfg.top_k, db_path=cfg.db_path)
        self.docdb = DocDB(cfg.db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), cfg.db_path))
        self.cur = con.cursor()
    
    def __call__(self, question):
        retrieved_data = super().__call__([{ 'question' : question }])
        candidate_passages = retrieved_data[0]['candidate_passages']
        contexts = []
        for doc_id, _, _ in candidate_passages:
            context = self.cur.execute("SELECT text FROM documents WHERE id = ?", (str(doc_id), )).fetchone()
            assert context != None
            context = word_tokenize(text_normalize(context[0]), format='text')
            contexts.append(context)
        return contexts


def get_negative_samples(bm25_retriever, question, positive_context, answer):
    negative_samples = []
    negative_contexts = bm25_retriever(question)
    for step, negative_context in enumerate(negative_contexts):
        if positive_context in negative_context: continue
        negative_samples.append({
            "context": negative_context,
            "question": question,
            "answer_text": '',
            "answer_start_idx": 0,
        })
    return negative_samples


def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def handle_file(data_path, bm25_retriever=None, debug_mode=False):
    with open(data_path, 'r', encoding='utf-8') as file_read:
        qa_data = json.load(file_read)['data']
    norm_samples = []
    error = 0

    print("Start parsing data...")
    for step, item in tqdm(enumerate(qa_data), total=len(qa_data), desc="Chunk {}".format(data_path)):
        if debug_mode and step == 10: break

        raw_context = item['text']
        raw_question = item['question']
        question = word_tokenize(text_normalize(raw_question), format='text')

        if item['category'] == 'FULL_ANNOTATION':
            short_candidate = item['short_candidate']
            short_candidate_start = item['short_candidate_start']
            wiki_answer = item['answer']

            prev_context = strip_context(raw_context[:short_candidate_start])
            answer       = strip_answer_string(short_candidate)
            post_context = strip_context(raw_context[short_candidate_start + len(answer):])

            prev_context = word_tokenize(text_normalize(prev_context), format='text')
            answer       = word_tokenize(text_normalize(answer), format='text')
            post_context = word_tokenize(text_normalize(post_context), format='text')

            context = "{} {} {}".format(prev_context, answer, post_context).strip()
            norm_samples.append({
                "context": context,
                "question": question,
                "answer_text": answer,
                "answer_start_idx": len("{} {}".format(prev_context, answer).strip()) - len(answer),
            })

            if bm25_retriever is not None:
                negative_samples = get_negative_samples(bm25_retriever, question, context, answer)
                norm_samples.extend(negative_samples)

        # elif item['category'] == 'FALSE_LONG_ANSWER':
        #     context  = word_tokenize(text_normalize(context), format='text')
        #     norm_samples.append({
        #         "context": context,
        #         "question": question,
        #         "answer_text": '',
        #         "answer_start_idx": 0,
        #     })

    print(f"Parsing completed!")
    return norm_samples
    

def preprocess(cfg, debug_mode=False):
    if not debug_mode and os.path.isfile(cfg.mrc_path):
        print('%s already exists! Not overwriting.' % cfg.mrc_path)
        return

    bm25_retriever = _BM25Retriever(cfg) if cfg.use_bm25 else None
    dict_data_squad = handle_file(data_path=cfg.data_path, bm25_retriever=bm25_retriever, debug_mode=debug_mode)
    with open(cfg.mrc_path, 'w', encoding='utf-8') as file:
        for item in dict_data_squad:
            file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))
    
    print("Total: {} samples".format(len(dict_data_squad)))


if __name__ == "__main__":
    # debugging purpose
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='qatask/database/datasets/train_test_files/train_sample.json')
    parser.add_argument("--mrc-path", type=str, default='qatask/database/datasets/data_for_finetuning/mrc_format_file.jsonl')
    parser.add_argument("--use-bm25", action='store_true')
    parser.add_argument("--db-path", type=str, default='qatask/database/wikipedia_db/wikisqlite.db')
    parser.add_argument("--index-path", type=str, default='checkpoint/indexes/BM25')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--debug-mode", action='store_true')
    args = parser.parse_args()

    if args.debug_mode:
        args.mrc_path = args.mrc_path[:-6] + '_debug.jsonl'
    preprocess(cfg=args, debug_mode=args.debug_mode)