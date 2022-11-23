from .base import BasePostProcessor
from pyserini.search.lucene import LuceneSearcher
from qatask.retriever.tfidf.doc_db import DocDB
import os
import os.path as osp
import sqlite3
from tqdm import tqdm
from fuzzywuzzy import process
import string
import nltk
from collections import Counter
from transformers import pipeline
import itertools
from torch.utils.data import DataLoader, Dataset
import numpy as np
import string

from .post_utils import handleReaderAns,  matching, matching_nospace, select_nearest, select_nearsest_shortest_withspace, checktype, date_transform

from collections import Counter
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def most_frequent(List, k):
    occurence_count = Counter(List)
    return occurence_count.most_common(k)

class BM25PostProcessor(BasePostProcessor):
    def __init__(self, cfg=None, db_path=None):
        super().__init__(cfg, db_path)
        self.denoisy = 5
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')
        self.cfg = cfg
        self.docdb = DocDB(db_path)
        self.top_k = cfg.top_k
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
        # self.pipeline = pipeline('question-answering', 
        #                         # device="cuda:0",
        #                         model='checkpoint/pretrained_model/checkpoint-3906',
        #                         tokenizer='checkpoint/pretrained_model/checkpoint-3906')

    def save_question(self, question, mode):             
        if mode == "test":
            if 'candidate_wikipages' in question.keys():
                question.pop('candidate_wikipages', None)
            if 'candidate_passages' in question.keys():
                question.pop('candidate_passages', None)
            if 'scores' in question.keys():
                question.pop('scores', None)
            if 'passage_scores' in question.keys():
                question.pop('passage_scores', None)
            if 'ans_type' in question.keys():
                    question.pop('ans_type', None)
            if type(question['answer']) is not str:
                question['answer'] = question['answer'][0]
        return question

    def merge(self, item):
      res_item = {
        'id': item['id'],
        'question': item['question'],
        'scores': [],
        'passage_scores': [],
        'candidate_passages': [],
        'answer': [],
        'candidate_wikipages': item['candidate_wikipages'],
      }
      answers_dict = {}
      for idx, ans in enumerate(item['answer']):
        if ans in answers_dict.keys():
            answers_dict[ans]['scores'] += item['scores'][idx]
        else:
          answers_dict[ans] = {
            "answer": ans,
            "scores": item['scores'][idx],
            "passage_scores": item['passage_scores'][idx],
            "candidate_passages": item["candidate_passages"][idx]
          }
      for _, ans in answers_dict.items():
        res_item['scores'].append(ans['scores'])
        res_item['answer'].append(ans['answer'])
        res_item['candidate_passages'].append(ans['candidate_passages'])
        res_item['passage_scores'].append(ans['passage_scores'])
      return res_item

    def find_matched_wikipage(self, reader_ans):
        macthed_list = []
        hits = self.searcher.search(reader_ans, self.top_k)
        doc_ids = [hit.docid for hit in hits]
        for doc_id in doc_ids:
            res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (doc_id, ))
            wikipage = res.fetchone()
            macthed_list.append(wikipage[0])
        return macthed_list

    def process(self, data, mode):
        print("Postprocessing...")
        emp = {'data': []}
        for question in tqdm(data["data"]):
            if question['ans_type'] > 0:
                anstype = question['ans_type']  
                if anstype == 3:
                    question['answer'] = None
                elif anstype == 2:
                    post_ans = date_transform(question['answer'], question['question'])
                    if mode == "val":
                        question['answer'] = [post_ans]
                    else:
                        question['answer'] = post_ans.strip()
                elif anstype == 1:
                    post_ans = ""
                    for d in reader_ans.lower().translate(str.maketrans('','',string.punctuation)).split():
                        if d.isnumeric():
                            post_ans = d
                    if mode == "val":
                        question['answer'] = [post_ans.strip()]
                    else:
                        question['answer'] = post_ans.strip()
            else:
                matched_wiki_answers = []
                tmp_reader_ans = [] # Debuging purpose
                for idx, reader_ans in enumerate(question['answer']):
                    post_ans = None
                    # TODO: Handle reader_ans is None type
                    reader_ans = handleReaderAns(reader_ans, question['question'])

                    tmp_reader_ans.append(reader_ans) # debugging purpose
                    # Extends current retrieved wikipages by new matched wiki entities, searched by reader_ans
                    candidate_wikipages = question['candidate_wikipages'].copy() + self.find_matched_wikipage(reader_ans)
                    choices = ["",] + [wikipage[5:].replace("_", " ") for wikipage in candidate_wikipages] # if we can not match by process extract -> default is an empty string
                    if reader_ans in choices:
                        wiki_with_space = reader_ans
                    else:
                        wiki_with_space = process.extractOne(reader_ans, choices)[0]
                    if wiki_with_space == "":
                        matched_wikipage = None
                        question['scores'][idx] = question['passage_scores'][idx] = 0
                    else:
                        matched_wikipage = 'wiki/' + wiki_with_space.replace(" ", "_")
                    matched_wiki_answers.append(matched_wikipage)

                # Sort answers, canidate passaegs by scores -> TODO: Shorter implementation lambda function in separated function
                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x] + question['passage_scores'][x], reverse=True)
                question['answer'] = [matched_wiki_answers[id] for id in ids_sorted]
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted]
                question['scores'] = [question['scores'][id] for id in ids_sorted]
                question['passage_scores'] = [question['passage_scores'][id] for id in ids_sorted]
                # Removing dupplicated answers and mergin their scores, notice "" is not a valid answer
                cp_qs = question.copy()
                question = self.merge(cp_qs)
                
                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x], reverse=True)
                
                # question['pos_answer'] = [question['answer'][id] for id in ids_sorted][:self.denoisy]
                question['answer'] = [question['answer'][id] for id in ids_sorted][:1]
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted][:self.denoisy]
                
                # For easy reading json logs
                question['scores'] = [question['scores'][id] for id in ids_sorted][:self.denoisy]
                question['passage_scores'] = [question['passage_scores'][id] for id in ids_sorted][:self.denoisy]

                # if mode == "val":
                # question['original_answers'] = tmp_reader_ans # debugging purpose
                # question['according_wikipages'] = matched_wiki_answers # debugging purpose

                # 1 line below is for baseline
                emp['data'].append(self.save_question(question, mode))
                continue

                unique_candidates = {}
                for idx, wikipage in enumerate(question['answer']):
                    if wikipage not in unique_candidates.keys():
                        unique_candidates[wikipage] = idx
                import itertools
                tuple_candidates = list(itertools.combinations(unique_candidates.keys(), 2))
                concat_passages = []
                for x,y in tuple_candidates:
                    passage1 = question["candidate_passages"][unique_candidates[x]]
                    passage2 = question["candidate_passages"][unique_candidates[y]]
                    concat_passage = passage1 + ". " + passage2
                    concat_passage_reverse = passage2 + ". " + passage1
                    concat_passages.append(concat_passage)
                    concat_passages.append(concat_passage_reverse)
                
                if len(tuple_candidates) == 0:
                    if mode == "val":
                        question['answer'] = [question['answer'][0]]
                    else:
                        question['answer'] = question['answer'][0]
                        question['answer'] = self.save_question(question['answer'], mode)
                    emp['data'].append(question)
                    continue
                if mode == 'val':
                    question["concat_passages"] = concat_passages
                prepared = [{"question": question["question"], "context": concat_passage} for concat_passage in concat_passages]
                prepared_dataset = ListDataset(prepared)
                predicted = []
                for batch in DataLoader(prepared_dataset, batch_size=30):
                    predicted_batch = self.pipeline(batch)
                    predicted.extend(predicted_batch)
                ans_concat = []
                for QA in predicted:
                    ans_concat.append((QA["answer"], QA["score"])) 
                if mode == "val":
                    question["ans_concat"] = ans_concat
                from collections import Counter
                def most_frequent(List, k):
                    occurence_count = Counter(List)
                    return occurence_count.most_common(k)
                final_ans_concat = []
                for ans, score in ans_concat:
                    ans = ans.translate(str.maketrans('','',string.punctuation))
                    wiki_ans = select_nearest(ans, question['answer'])
                    final_ans_concat.append((wiki_ans, score))

                if mode == "val":
                    question["answer_concat"] = final_ans_concat
                
                record = {}
                for matched_wiki_answers, score in final_ans_concat:
                    if matched_wiki_answers not in record.keys():
                        record[matched_wiki_answers] = score
                    else:
                        record[matched_wiki_answers] += score

                question['answer'] = [max(record, key=record.get)]
            
            emp['data'].append(self.save_question(question, mode))
        return emp

    def __call__(self, data, mode):
        return self.process(data, mode)
