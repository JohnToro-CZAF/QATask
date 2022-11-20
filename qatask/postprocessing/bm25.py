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

    def process(self, data, mode):
        print("Postprocessing...")
        for question in tqdm(data["data"]):
            final_ans = []
            for reader_ans in question['answer']:
                post_ans = None
                if reader_ans is None:
                    if mode == "test":
                        if question['candidate_wikipages'] != []:
                            question.pop('candidate_wikipages', None)
                    continue
                anstype = self.checktype(reader_ans, question['question'])
                if anstype > 0:
                    if anstype == 3:
                        pass
                    elif anstype == 2:
                        post_ans = self.date_transform(reader_ans, question['question'])
                        post_ans = reader_ans.strip()
                    elif anstype == 1:
                        post_ans = ""
                        for d in reader_ans.lower().translate(str.maketrans('','',string.punctuation)).split():
                            if d.isnumeric():
                                post_ans = d
                        post_ans = post_ans.strip()
                else:
                    hits = self.searcher.search(reader_ans, self.top_k)
                    doc_ids = [hit.docid for hit in hits]
                    # for hit in hits:
                    #     print(hit.raw)
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
                    if 'tỉnh' in query:
                        query = query.replace('tỉnh', '')
                    if 'Tỉnh' in query:
                        query = query.replace('Tỉnh', '')

                    try:
                        wikipage = process.extractOne(query, choices)[0]
                        # print(wikipage)
                        post_ans = 'wiki/' + wikipage.replace(" ", "_")
                    except:
                        print("can not retrieve this question wikipage")

                # For the test mode. Validation mode needs
                final_ans.append(post_ans)

            if len(final_ans) > 1:
                # Selecting the max score after postprocessing
                id = 0
                for idx, ans in enumerate(final_ans):
                    if question['scores'][idx] + question['passage_scores'][idx] > question['scores'][id] + question['passage_scores'][id]:
                        id = idx
                ids_sorted = sorted(range(len(question['scores'])),key=lambda x: question['scores'][x] + question['passage_scores'][x], reverse=True)
                # print(ids_sorted)
                # print(final_ans)
                # print(question['scores'])
                # print(question['passage_scores'])
                question['answer'] = [final_ans[id] for id in ids_sorted][:15]
                # print(question['answer'])
                question['answer'] =  [max(set(question['question']), key = question['question'].count)]
            else:
                question['answer'] = final_ans[0]

            if mode == "test":
                if question['candidate_wikipages'] != []:
                    question.pop('candidate_wikipages', None)

        return data

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