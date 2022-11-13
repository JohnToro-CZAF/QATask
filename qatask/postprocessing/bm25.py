from .base import BasePostProcessor
from pyserini.search.lucene import LuceneSearcher
from qatask.retriever.tfidf.doc_db import DocDB
import os
import os.path as osp
import sqlite3
from tqdm import tqdm
from fuzzywuzzy import process
import string

class BM25PostProcessor(BasePostProcessor):
    def __init__(self, cfg, db_path):
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
        if "bao nhiêu" in question:
            for d in words:
                if d.isdigit():
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
        lookup = {'năm': '', 'tháng': '', 'ngày': ''}
        for w in words:
            pass
        for idx, w in enumerate(words):
            if w in lookup and idx+1 < len(words):
                if(words[idx+1].isdigit()):
                    lookup[w] = words[idx+1]
        ans = ""
        if 'năm nào' in question:
            ans += 'năm ' + lookup['năm'] + " "
            return ans
        if lookup['ngày'] != "":
            ans += 'ngày ' + lookup['ngày'] + " "
        if lookup['tháng'] != "":
            ans += 'tháng ' + lookup['tháng'] + " "
        if lookup['năm'] != "":
            ans += 'năm ' + lookup['năm'] + " "
        ans = ans.strip()
        return ans

    def process(self, data):
        print("Postprocessing...")
        for question in tqdm(data["data"]):
            anstype = self.checktype(question['answer'], question['question'])
            if anstype > 0:
                if anstype == 3:
                    question['answer'] = 'null'
                elif anstype == 2:
                    question['answer'] = self.date_transform(question['answer'], question['question'])
                elif anstype == 1:
                    tmpans = None
                    for d in question['answer']:
                        if d.isdigit():
                            tmpans = d
                    question['answer'] = d
                continue
            hits = self.searcher.search(question['answer'])
            try:
                # doc_id = hits[0].docid
                # res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                # wikipage = res.fetchone()
                # TODO: There are many canidate wikipage -> find the one who is 
                # nearest
                doc_ids = []
                for i in range(0, self.top_k):
                    doc_ids.append(hits[i].docid)
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (doc_ids, ))
                wikipages = res.fetchall()
                choices = [wikipage[0][5:].replace("_", " ") for wikipage in wikipages]
                wikipage = process.extractOne(question['answer'], choices)[0]
                question['answer'] = 'wiki/' + wikipage.replace(" ", "_")
                question['answer'] = wikipage
            except:
                print("can no retrieve this question wikipage")
             
        return data

    def __call__(self, data):
        return self.process(data)

def main():
    processor = BasePostProcessor()
    text = "ngày 1 tháng 2 năm 2019"
    print(processor.date_transform(text))
if __name__ == '__main__':
    main()