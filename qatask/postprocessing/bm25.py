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
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import itertools
from torch.utils.data import DataLoader, Dataset
import numpy as np
import string
import itertools
import time
import pickle
import sys
# sys.path.append("GENRE")

# from genre.fairseq_model import mGENRE
# from genre.trie import Trie, MarisaTrie

from .post_utils import handleReaderAns,  matching, matching_nospace, select_nearest, select_nearsest_shortest_withspace, checktype, date_transform

from collections import Counter
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

def most_frequent(List, k):
    occurence_count = Counter(List)
    return occurence_count.most_common(k)

class BM25PostProcessor(BasePostProcessor):
    def __init__(self, cfg=None, db_path=None):
        super().__init__(cfg, db_path)
        self.cfg = cfg
        self.concat_threshold = self.cfg.concat_threshold
        self.denoisy = self.cfg.denoisy
        self.top_k = cfg.top_k
        
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    
        self.pipeline = pipeline('question-answering', 
                                device="cuda:0",
                                model='nguyenvulebinh/vi-mrc-large',
                                tokenizer='nguyenvulebinh/vi-mrc-large')
        self.pipeline2 = pipeline('question-answering',
                            model="hogger32/xlmRoberta-for-VietnameseQA",
                            tokenizer="hogger32/xlmRoberta-for-VietnameseQA",
                            device="cuda:0")
        self.pipeline3 = pipeline('question-answering',
                                  model="checkpoint/pretrained_model/electra/checkpoint-19000",
                                  tokenizer="checkpoint/pretrained_model/electra/checkpoint-19000",
                                  device="cuda:0")
        self.linker_tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
        self.linker_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval().to('cuda:0')
        # self.linker_trie = 
        # self.model = mGENRE.from_pretrained("checkpoint/data_linker/fairseq_multilingual_entity_disambiguation").eval()


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
            # Heuristic by combining scores seems ineffective ->
            # answers_dict[ans]['scores'] = max(item['scores'][idx], answers_dict[ans]['scores'])
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

    def linking_to_wiki(self, reader_answer, formated_ctx):
        # Return to fine grained wikipage
        output = self.linker_model.generate(
            **self.linker_tokenizer(formated_ctx, return_tensors="pt").to('cuda:0'),
            num_beams=1,
            num_return_sequences=1,
            max_length = 200
        )
        fine_grained_ans = self.linker_tokenizer.batch_decode(output, skip_special_tokens=True)[0][:-5]
        return fine_grained_ans.strip()
    
    def linking_to_wiki_batches(self, formated_ctxs):
        # Return to fine grained wikipage
        batch_size = 25
        output = []
        for batch in split(formated_ctxs, batch_size):
            print(len(batch))
            # output_batch = [ctx[0]['text'] for ctx in output_batch]
            output_batch = self.linker_model.generate(
                **self.linker_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=250).to('cuda:0'),
                num_beams=1,
                num_return_sequences=1,
                max_length = 250
            )
            output.extend(output_batch)
        fine_grained_ans = self.linker_tokenizer.batch_decode(output, skip_special_tokens=True)
        return [a[:-5].strip() for a in fine_grained_ans]

    def process_ans_type(self, question, mode):
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
            for d in question['answer'].lower().translate(str.maketrans('','',string.punctuation)).split():
                if d.isnumeric():
                    post_ans = d
            if mode == "val":
                question['answer'] = [post_ans.strip()]
            else:
                question['answer'] = post_ans.strip()
        return question
    
    def find_wikipage(self, question):
        matched_wiki_answers = []
        tmp_reader_ans = question['answer'].copy() # Debuging purpose
        reader_ans = [handleReaderAns(a, question['question']) for a in question['answer']]
        candidate_wikipages = [question['candidate_wikipages'].copy() + self.find_matched_wikipage(a) for a in reader_ans]
        choices = [["",] + [wikipage[5:].replace("_", " ") for wikipage in candidate_wikipages[i]] for i in range(len(reader_ans))]
        
        wiki_with_space_wiki_linked = self.linking_to_wiki_batches([question['question'] + ' ' + question['formated_passages'][i] for i in range(len(reader_ans))])
        wiki_with_space_linked = [process.extractOne(wiki_with_space, choice)[0] for wiki_with_space, choice in zip(wiki_with_space_wiki_linked, choices)]
        
        wiki_with_space = [ans if ans in choices[idx] else wiki_with_space_linked[idx] for idx, ans in enumerate(reader_ans)]
        for idx, wiki in enumerate(wiki_with_space):
            if wiki == "":
                matched_wikipage = None
                question['scores'][idx] = question['passage_scores'][idx] = 0
            else:
                matched_wikipage = 'wiki/' + wiki.replace(" ", "_")

            matched_wiki_answers.append(matched_wikipage)

        # for idx, reader_ans in enumerate(question['answer']):
        #     # TODO: Handle reader_ans is None type
        #     reader_ans = handleReaderAns(reader_ans, question['question'])

        #     tmp_reader_ans.append(reader_ans) # debugging purpose
        #     # Extends current retrieved wikipages by new matched wiki entities, searched by reader_ans
        #     candidate_wikipages = question['candidate_wikipages'].copy() + self.find_matched_wikipage(reader_ans)
        #     choices = ["",] + [wikipage[5:].replace("_", " ") for wikipage in candidate_wikipages] # if we can not match by process extract -> default is an empty string
        #     # TODO: New neural mode. Using fuzzy wuzzy does not solve the entity linking problem
        #     if reader_ans in choices:
        #         wiki_with_space = reader_ans
        #     else:
        #         wiki_with_space = self.linking_to_wiki(reader_ans, question['question'] + ' ' + question['formated_passages'][idx])
        #         wiki_with_space = process.extractOne(wiki_with_space, choices)[0]
        #     if wiki_with_space == "":
        #         matched_wikipage = None
        #         question['scores'][idx] = question['passage_scores'][idx] = 0
        #     else:
        #         matched_wikipage = 'wiki/' + wiki_with_space.replace(" ", "_")

        #     matched_wiki_answers.append(matched_wikipage)
        return question, matched_wiki_answers, tmp_reader_ans
    
    def reading_concat(self, question, concat_passages, mode):
        print("Have to re read", len(concat_passages))
        ans_concat = []
        prepared = [{"question": question["question"], "context": concat_passage} for concat_passage in concat_passages]
        prepared_dataset = ListDataset(prepared)
        # 1st Reader
        predicted = []
        for batch in DataLoader(prepared_dataset, batch_size=30):
            predicted_batch = self.pipeline(batch)
            predicted.extend(predicted_batch)
        for QA in predicted:
            ans_concat.append((QA["answer"], QA["score"]))
        # 2nd Reader
        predicted = []
        for batch in DataLoader(prepared_dataset, batch_size=30):
            predicted_batch = self.pipeline2(batch)
            predicted.extend(predicted_batch)
        for QA in predicted:
            ans_concat.append((QA["answer"], QA["score"]))
        # 3rd Reader
        predicted = []
        for batch in DataLoader(prepared_dataset, batch_size=30):
            predicted_batch = self.pipeline3(batch)
            predicted.extend(predicted_batch)
        for QA in predicted:
            ans_concat.append((QA["answer"], QA["score"]))

        if mode == "val":
            question["ans_concat"] = ans_concat
        final_ans_concat = []
        for ans, score in ans_concat:
            ans = ans.translate(str.maketrans('','',string.punctuation))
            # New generated answer from rereader is chosen from the choices question['answer'] 
            # which is linked from linker model and generated from bertbasereader
            # TODO: Link to answer by linker model (longer inferring time), heuristic select nearest seems not the best
            wiki_ans = select_nearest(ans, question['answer'])
            final_ans_concat.append((wiki_ans, score))

        if mode == "val":
            question["answer_concat"] = final_ans_concat
            question["concat_passages"] = concat_passages
    
        record = {}
        for matched_wiki_answers, score in final_ans_concat:
            if matched_wiki_answers not in record.keys():
                record[matched_wiki_answers] = score
            else:
                record[matched_wiki_answers] += score
        if mode == "val":
            question['answer'] = [max(record, key=record.get)]
        else:
            question['answer'] = max(record, key=record.get)
        
        return question
    
    def building_concat_passages(self, question):
        unique_candidates = {}
        best_scores = 0
        for idx, wikipage in enumerate(question['answer']):
            if wikipage not in unique_candidates.keys() and (abs(question['scores'][idx] - best_scores) < self.concat_threshold or best_scores == 0):
                best_scores = question['scores'][idx]
                unique_candidates[wikipage] = idx
        tuple_candidates = list(itertools.combinations(unique_candidates.keys(), 2))
        concat_passages = []
        for x,y in tuple_candidates:
            passage1 = question["candidate_passages"][unique_candidates[x]]
            passage2 = question["candidate_passages"][unique_candidates[y]]
            concat_passage = passage1 + ". " + passage2
            concat_passage_reverse = passage2 + ". " + passage1
            concat_passages.append(concat_passage)
            concat_passages.append(concat_passage_reverse)
        return tuple_candidates, concat_passages
        
    def process(self, data, mode):
        print("Postprocessing...")
        emp = {'data': []}
        for question in tqdm(data["data"]):
            if question['ans_type'] > 0:
                question = self.process_ans_type(question, mode)
            else:
                time_ind = time.time()
                question, matched_wiki_answers, tmp_reader_ans = self.find_wikipage(question)
                print("Find wikipage costs: ", time.time() - time_ind)
                # Sort answers, canidate passaegs by scores -> TODO: Shorter implementation lambda function in separated function
                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x] + question['passage_scores'][x], reverse=True)
                print(len(ids_sorted))
                question['answer'] = [matched_wiki_answers[id] for id in ids_sorted]
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted]
                question['scores'] = [question['scores'][id] for id in ids_sorted]
                question['passage_scores'] = [question['passage_scores'][id] for id in ids_sorted]

                # Removing dupplicated answers and merging their scores, notice "" is not a valid answer
                cp_qs = question.copy()
                question = self.merge(cp_qs)
                
                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x], reverse=True)        
                # question['pos_answer'] = [question['answer'][id] for id in ids_sorted][:self.denoisy]
                question['answer'] = [question['answer'][id] for id in ids_sorted][:self.denoisy]
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted][:self.denoisy]
                
                # For easy reading json logs
                question['scores'] = [question['scores'][id] for id in ids_sorted][:self.denoisy]
                question['passage_scores'] = [question['passage_scores'][id] for id in ids_sorted][:self.denoisy]

                if mode == "val":
                    question['original_answers'] = [tmp_reader_ans[id] for id in ids_sorted] # debugging purpose
                    question['according_wikipages'] = [matched_wiki_answers[id] for id in ids_sorted] # debugging purpose

                # 1 line below is for baseline
                # emp['data'].append(self.save_question(question, mode))
                # continue
                tuple_candidates, concat_passages = self.building_concat_passages(question)
                if len(tuple_candidates) == 0:
                    if mode == "val":
                        question['answer'] = [question['answer'][0]]
                    else:
                        question['answer'] = question['answer'][0]
                    emp['data'].append(self.save_question(question, mode))
                    continue
                question = self.reading_concat(question, concat_passages, mode)

            emp['data'].append(self.save_question(question, mode))
        return emp

    def __call__(self, data, mode):
        return self.process(data, mode)

class SimplePostProcessor(BM25PostProcessor):
    def __init__(self, cfg=None, db_path=None):
        super().__init__(cfg, db_path)
    
    def find_wikipage(self, question):
        matched_wiki_answers = []
        tmp_reader_ans = [] # Debuging purpose
        #Only one answer
        for idx, reader_ans in enumerate(question['answer']):
            post_ans = None
            # TODO: Handle reader_ans is None type
            reader_ans = handleReaderAns(reader_ans, question['question'])

            tmp_reader_ans.append(reader_ans) # debugging purpose
            # Extends current retrieved wikipages by new matched wiki entities, searched by reader_ans
            candidate_wikipages = question['candidate_wikipages'].copy() + self.find_matched_wikipage(reader_ans)
            choices = ["",] + [wikipage[5:].replace("_", " ") for wikipage in candidate_wikipages] # if we can not match by process extract -> default is an empty string
            # TODO: New neural mode. Using fuzzy wuzzy does not solve the entity linking problem
            if reader_ans in choices:
                wiki_with_space = reader_ans
            else:
                wiki_with_space = self.linking_to_wiki(reader_ans, question['question'] + ' ' + question['formated_passages'][idx])
                wiki_with_space = process.extractOne(wiki_with_space, choices)[0]
            if wiki_with_space == "":
                matched_wikipage = None
                question['scores'][idx] = question['passage_scores'][idx] = 0
            else:
                matched_wikipage = 'wiki/' + wiki_with_space.replace(" ", "_")
            matched_wiki_answers.append(matched_wikipage)
            return question, matched_wiki_answers, tmp_reader_ans
    
    def process(self, data, mode):
        print("Postprocessing...")
        emp = {'data': []}
        for question in tqdm(data["data"]):
            if question['ans_type'] > 0:
                question = self.process_ans_type(question, mode)
            else:
                question, matched_wiki_answers, tmp_reader_ans = self.find_wikipage(question)
            emp['data'].append(self.save_question(question, mode))
        
        return emp
