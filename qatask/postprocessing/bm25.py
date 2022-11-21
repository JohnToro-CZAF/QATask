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

def matching(short_form: str, wikipage: str):
    match = 0
    words_wiki = wikipage[5:].replace('_', ' ').split()
    words_short_form = short_form.split()
    decay = 0.5
    constant = 1
    for w in words_short_form:
        if w in words_wiki:
            match += constant
            constant *= decay
    return match

def matching_nospace(short_form: str, wikipage: str):
    match = 0
    words_wiki = wikipage.split()
    words_short_form = short_form.split()
    decay = 0.5
    constant = 1
    for w in words_short_form:
        if w in words_wiki:
            match += constant
            constant *= decay
    return match

def select_nearest(short_form: str, wikipages):
    id = 0
    lst = 0
    for idx, wiki in enumerate(wikipages):
        if matching(short_form, wiki) > lst:
            id = idx
            lst = matching(short_form, wiki)
    return wikipages[id]

def select_nearsest_shortest_withspace(short_form:str, wikipages):
    mx = 0
    pos = []
    # print(short_form, wikipages)
    for wiki in wikipages:
        # print(matching_nospace(short_form, wiki))
        if matching_nospace(short_form, wiki) > mx:
            mx = matching_nospace(short_form, wiki) 
    # print(mx)
    for wiki in wikipages:
        # print(matching_nospace(short_form, wiki))
        if matching_nospace(short_form, wiki) == mx:
            pos.append(wiki)
    # print(pos)
    return min(pos, key=lambda x: len(x))

class BM25PostProcessor(BasePostProcessor):
    def __init__(self, cfg=None, db_path=None):
        super().__init__(cfg, db_path)
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')
        self.cfg = cfg
        self.docdb = DocDB(db_path)
        self.top_k = cfg.top_k
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
        self.pipeline = pipeline('question-answering', 
                                #  model='nguyenvulebinh/vi-mrc-large', 
                                #  tokenizer='nguyenvulebinh/vi-mrc-large', 
                                model='checkpoint/pretrained_model/checkpoint-3906',
                                tokenizer='checkpoint/pretrained_model/checkpoint-3906',
                                 device="cuda:0")
    
    def checktype(self, text, question):
        text = text.lower().translate(str.maketrans('','',string.punctuation))
        words = text.split()
        """Check if the text is a date or a number."""
        time_indicators = ["năm nào", "năm mấy", "năm bao nhiêu", "ngày bao nhiêu", "ngày tháng năm nào", "thời điểm nào", "ngày tháng âm lịch nào hằng năm", "thời gian nào", "lúc nào", "giai đoạn nào trong năm"]
        if any(idc in question.lower() for idc in time_indicators):
            return 2
        if "có bao nhiêu" in question:
            for d in words:
                if d.isnumeric():
                    return 1
        if text == "":
            return 3
        words = text.split()
        for w in words:
            if w == 'năm' or w == 'tháng' or w == 'ngày':
                return 2
        if len(words) == 1 and words[0].isdigit():
            return 1
        return 0

    def date_transform(self, text, question):
        text = text.lower().translate(str.maketrans('','',string.punctuation))
        words = text.split()
        lookup = {'năm': '', 'tháng': '', 'ngày': '', 'mùng': ''}
        for idx, w in enumerate(words):
            if w in lookup and idx+1 < len(words):
                if(words[idx+1].isnumeric()):
                    lookup[w] = words[idx+1]
        ans = ""
        lisw = ["ngày", "tháng", "năm"]
        lisq = []
        for w in lisw:
            if w in question:
                lisq.append(w)
        if len(lisq) == 3 or len(lisq) == 2:
            # day and month or full day -> Only take what it asked for
            for w in lisq:
                if lookup[w] != "":
                    ans += w + " " + lookup[w] + " "
                elif w == 'ngày' and lookup['mùng'] != "":
                    ans += 'mùng' + ' ' + lookup["mùng"] + " "
        elif len(lisq) == 1:
            if(lisq[0] == "năm"):
                if lookup[lisq[0]] != "":
                    ans += lisq[0] + " " + lookup[lisq[0]]
                    return ans
                else:
                    # There is no năm inside the answer text, have to search for literal
                    for d in words:
                        if d.isnumeric():
                            return "năm" + " " + d
            elif(lisq[0] == "tháng"):
                if lookup[lisq[0]] != "":
                    ans += lisq[0] + " " + lookup[lisq[0]]
                    return ans
                else :
                    for d in words:
                        if d.isnumeric():
                            return "tháng" + " " + d
            # question on day -> full day
            for w in lisw:
                if lookup[w] != "":
                    ans += w + " " + lookup[w] + " "
                elif w == 'ngày' and lookup['mùng'] != "":
                    ans += 'mùng' + ' ' + lookup["mùng"] + " "
        else:
            # Asking a date but there is no indicator month, year, or day -> full day
            for w in lisw:
                if lookup[w] != "":
                    ans += w + " " + lookup[w] + " "
                elif w == 'ngày' and lookup['mùng'] != "":
                    ans += 'mùng' + ' ' + lookup["mùng"] + " "
        return ans.strip()

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
    #   print(res_item['scores'], res_item['answer'])
      return res_item

    def process(self, data, mode):
        print("Postprocessing...")
        emp = {'data': []}
        for question in tqdm(data["data"]):
            if question['ans_type'] > 0:
                anstype = question['ans_type']  
                if anstype == 3:
                    question['answer'] = None
                elif anstype == 2:
                    post_ans = self.date_transform(question['answer'], question['question'])
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
                # emp['data'].append(question)
            else:        
                final_ans = []
                for idx, reader_ans in enumerate(question['answer']):
                    post_ans = None
                    if reader_ans is None:
                        # if mode == "test":
                        #     if question['candidate_wikipages'] != []:
                        #         question.pop('candidate_wikipages', None)
                        continue
                    # print('-'*100)
                    # print("Reader answer: ", reader_ans)
                    reader_ans = reader_ans.translate(str.maketrans('','',string.punctuation))
                    hits = self.searcher.search(reader_ans, self.top_k)
                    doc_ids = [hit.docid for hit in hits]
                    wikipages = question['candidate_wikipages'].copy()
                    for doc_id in doc_ids:
                        res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (doc_id, ))
                        wikipage = res.fetchone()
                        wikipages.append(wikipage[0])
                    choices = [wikipage[5:].replace("_", " ") for wikipage in wikipages]
                    query = reader_ans
                    query = query.replace('_', " ").strip()
                    if not query.startswith("."):
                        query = query.translate(str.maketrans('','',string.punctuation))
                        query.strip()
                    choices.insert(0, "")
                    # print(query, '-------', choices)
                    if query in choices:
                        wikipage = query
                    else:
                        # print(query, '-------', choices)
                        # try:
                        wikipage = process.extractOne(query, choices)[0]
                        # except:
                            # wikipage = ""
                        # print("answer: ", wikipage)
                    # print("answer: ", wikipage)
                    if wikipage == "":
                        post_ans = None
                        question['scores'][idx] = 0
                        question['passage_scores'][idx] = 0
                    else:
                        post_ans = 'wiki/' + wikipage.replace(" ", "_")
                    final_ans.append(post_ans)

                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x] + question['passage_scores'][x], reverse=True)

                question['answer'] = [final_ans[id] for id in ids_sorted]
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted]
                question['scores'] = [question['scores'][id] for id in ids_sorted]
                question['passage_scores'] = [question['passage_scores'][id] for id in ids_sorted]

                cp_qs = question.copy()
                question = self.merge(cp_qs)
                # print(question['answer'], question['scores'])

                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x], reverse=True)
                # print("before"*20)
                # print(question)
                denoisy = 2
                # Huy's code
                # print(question['answer'], question['scores'])
                original_ans = [question["answer"][id] for id in ids_sorted]
                if mode == "val":
                    question['original_ans'] = original_ans
                    question['candidate_answer'] = [question['answer'][id] for id in ids_sorted]
                
                question['answer'] = [question['answer'][id] for id in ids_sorted][:denoisy]
                # 1 line below is for baseline
                emp['data'].append(self.save_question(question, mode))
                continue
                question['candidate_passages'] = [question['candidate_passages'][id] for id in ids_sorted][:denoisy]
                unique_candidates = {}
                print(question['answer'])
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
                    # final_ans_concat.append((wiki_ans, score))
                    final_ans_concat.append((wiki_ans, score))
                if mode == "val":
                    question["answer_concat"] = final_ans_concat
                record = {}
                for final_ans, score in final_ans_concat:
                    if final_ans not in record.keys():
                        record[final_ans] = score
                    else:
                        record[final_ans] += score

                # most_common = most_frequent(final_ans_concat, 5)
                # print(most_common)
                if mode == "val":
                    # question["answer"] = [m[0] for m in most_common]
                    # question['answer'] = [most_common[0][0]]
                    question['answer'] = [max(record, key=record.get)]
                else:
                    # question["answer"] = most_common[0][0]
                    question['answer'] = max(record, key=record.get)
                    # End's huy code
            # print('after'*20)
            # print(question)
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
            emp['data'].append(question)
        return emp

    def __call__(self, data, mode):
        return self.process(data, mode)

def checktype(text, question):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    words = text.split()
    """Check if the text is a date or a number."""
    if "năm" or "ngày" or "tháng" in question:
        return 2
    if "bao nhiêu" in question:
        for d in words:
            if d.isnumeric():
                return 1
    if text == "":
        return 3
    words = text.split()
    for w in words:
        if w == 'năm' or w == 'tháng' or w == 'ngày':
            return 2
    if len(words) == 1 and words[0].isdigit():
        return 1
    return 0

def date_transform(text, question):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    words = text.split()
    lookup = {'năm': '', 'tháng': '', 'ngày': ''}
    for idx, w in enumerate(words):
        if w in lookup and idx+1 < len(words):
            if(words[idx+1].isnumeric()):
                lookup[w] = words[idx+1]
    ans = ""
    lisw = ["ngày", "tháng", "năm"]
    lisq = []
    for w in lisw:
        if w in question:
            lisq.append(w)
    for w in lisq:
        if lookup[w] != "":
            ans += w + " " + lookup[w] + " "
    if ans == "":
        pref = ""
        if "năm" in question:
            pref = "năm"
        elif "tháng" in question:
            pref = "tháng"
        elif "ngày" in question:
            pref = "ngày"
        for d in words:
            if d.isnumeric():
                return pref + " " + d
    return ans.strip()

def main():
    text = "năm 1922"
    question =  "Cộng hòa Liên bang Nga hiện nay được thành lập năm nào"
    anstype = checktype(text, question)
    print(anstype)
    if anstype > 0:
        if anstype == 3:
            ans = ""
        elif anstype == 2:
            ans = date_transform(text, question)
            ans = ans.strip()
        elif anstype == 1:
            tmpans = ""
            for d in text.lower().translate(str.maketrans('','',string.punctuation)).split():
                if d.isnumeric():
                    tmpans = d
            ans = tmpans
            ans = ans.strip()
    else:
        print("Erroring")
    print("checkspace:",ans,"endspace")
    
if __name__ == '__main__':
    main()