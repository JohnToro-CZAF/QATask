from .base import BasePostProcessor
from pyserini.search.lucene import LuceneSearcher
from qatask.retriever.tfidf.doc_db import DocDB
import os
import os.path as osp
import sqlite3
from tqdm import tqdm

class BM25PostProcessor(BasePostProcessor):
    def __init__(self, cfg, db_path):
        super().__init__(cfg, db_path)
        self.searcher = LuceneSearcher(cfg.index_path)
        self.searcher.set_language('vn')
        self.cfg = cfg
        self.docdb = DocDB(db_path)
        con = sqlite3.connect(osp.join(os.getcwd(), db_path))
        self.cur = con.cursor()
    
    def process_year(self, ans):
        if "năm" or "Năm" in ans:
            return ans
        else:
            return None

    def process(self, data):
        print("Postprocessing...")
        for question in tqdm(data["data"]):
            # if self.process_year(question['answer']) is not None:
            #     question['answer'] = self.process_year(question['answer'])
            #     continue
            hits = self.searcher.search(question['answer'])
            try:
                doc_id = hits[0].docid
                res = self.cur.execute("SELECT wikipage FROM documents WHERE id = ?", (str(doc_id), ))
                wikipage = res.fetchone()
                print(wikipage)
                question['answer'] = wikipage
            except:
                print("can no retrieve this question wikipage")
             
        return data

    def __call__(self, data):
        return self.process(data)

class HoangPostProcessor(object):
    """Base class for post processors."""
 
    def __init__(self, cfg=None):
        """Initialize the post processor.
 
        Args:
            config (dict): Configuration for the post processor.
        """
        self.cfg = cfg
 
    def checktype(self, text):
        """Check if the text is a date or a number."""
        words = text.split()
        for w in words:
            if w == 'năm' or w == 'tháng' or w == 'ngày':
                return 2
        if len(words) == 1 and words[0].isdigit():
            return 1
        return 0
    def date_transform(self, text):
        words = text.split()
        lookup = {'năm': '', 'tháng': '', 'ngày': ''}
        for idx, w in enumerate(words):
            if w in lookup and idx+1 < len(words):
                if(words[idx+1].isdigit()):
                    lookup[w] = words[idx+1]
        ans = ""
        if lookup['ngày'] != "":
            ans += 'ngày ' + lookup['ngày'] + " "
        if lookup['tháng'] != "":
            ans += 'tháng ' + lookup['tháng'] + " "
        if lookup['năm'] != "":
            ans += 'năm ' + lookup['năm'] + " "
        áns = ans.strip()
        return ans
 
    def postprocess(self, text):
        if text == '':
            return 'null'
        anstype = self.checktype(text)
        if anstype == 2:
            return self.date_transform(text)
        elif anstype == 1:
            # Only number
            return text
        elif anstype == 0:
            return 'wiki/'+text.replace(" ", "_")
        else:
            assert("Error")
 
    def process(self, data):
        """Process the data.
 
        Args:
            data (dict): List of {question, answers:List, scores:List}
 
        Returns:
            dict: Processed data.
        """
        return self.postprocess(data)
 
    def __call__(self, data):
        return self.process(data)
def main():
    processor = BasePostProcessor()
    text = "ngày 1 tháng 2 năm 2019"
    print(processor(text))
if __name__ == '__main__':
    main()