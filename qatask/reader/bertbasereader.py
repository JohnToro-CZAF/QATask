from transformers import pipeline
from .base import BaseReader
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import string
import numpy as np
import re
from typing import List, Dict, Sequence, Tuple
from collections import defaultdict

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

def checktype(text, question):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    words = text.split()
    """Check if the text is a date or a number."""
    time_indicators = ["năm nào", "năm mấy", "năm bao nhiêu", "ngày bao nhiêu", "ngày tháng năm nào", "thời điểm nào", "ngày tháng âm lịch nào hằng năm", "thời gian nào", "lúc nào", "giai đoạn nào trong năm", "thời điểm", "Thời điểm"]
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

def check_dominant_type(answers: List[str], scores: List[float], question: str) -> int:
  cnt = defaultdict()
  for idx, answer in enumerate(answers):
    cnt[checktype(answer, question)] = cnt.get(checktype(answer, question), 0) + scores[idx]
  print(question, " ", max(cnt, key=cnt.get)) # Debugging
  return max(cnt, key=cnt.get)

class BertReader(BaseReader):
    # class __name__
    def __init__(self, cfg=None, tokenizer=None, db_path=None) -> None:
        super().__init__(cfg, tokenizer, db_path)
        self.cfg = cfg
        
        # 1st reader: RoBERTa
        self.pipeline = pipeline('question-answering', 
                                 model=self.cfg.model_checkpoint, 
                                 tokenizer=self.cfg.model_checkpoint, 
                                 device="cuda:0", 
                                 batch_size=self.cfg.batch_size)
        # 2nd reader: xlmRoBERTa
        self.pipeline2 = pipeline('question-answering',
                                  model="hogger32/xlmRoberta-for-VietnameseQA",
                                  tokenizer="hogger32/xlmRoberta-for-VietnameseQA",
                                  device="cuda:0")
        # 3rd reader: ELECTRA
        self.pipeline3 = pipeline('question-answering',
                                  model="checkpoint/pretrained_model/electra/checkpoint-19000",
                                  tokenizer="checkpoint/pretrained_model/electra/checkpoint-19000",
                                  device="cuda:0")
        # This number has to be smaller or equal to the self.top_k at retriever
        # Experiments has shown that if sieve_threshold < self.top_k did not bring any improvement results
        self.sieve_threshold = self.cfg.sieve_threshold

    def clean_ctx(self, ctx):
      pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\?|\!|\;|\=|\+|\*|\%|\$|\#|\@|\^|\&|\~|\`|\|')
      ctx = pattern.sub('', ctx)
      ctx = ctx.replace('_', ' ')
      #TODO: Instead of removing '-' solely just to solve the date problem then no need we have to consider other cases that need that '-' to be stayed. E.g wiki/Lockheed_SR-71_Blackbird
      # ctx = ctx.replace("-", " đến ")
      ctx = ctx.replace(":", " là ")
      ctx = ctx.replace("=", " bằng ")
      ctx = ctx.replace("\/", " ")
      return ctx

    def prepare(self, data):
        """
        Args:
          data: A list of {question, contexts:List, passage_scores: List}
        Returns:
          data: A list of {question, context}
        """
        res = []
        for item in data:
            ctxs = item['contexts']
            for context in ctxs:
                res.append({'question': item['question'], 'context': self.clean_ctx(context)})
        return res

    def postprocess(self, prepared, predicted):
      answer = []
      QAans = {'question': "",
              'scores':[], 
              'starts':[], 
              'end':[], 
              'answers':[]}
      prepared.append({'question': 'dummy', 'context': 'dummy'})
      for idx, QA in enumerate(predicted):
        question = prepared[idx]['question']
        QAans['question'] = question
        QAans['scores'].append(QA['score'])
        QAans['starts'].append(QA['start'])
        QAans['end'].append(QA['end'])
        QAans['answers'].append(QA['answer'])
        if len(predicted) == 1:
          answer.append(QAans)
        else:
            if prepared[idx]['question'] != prepared[idx+1]['question']:
              answer.append(QAans)
              QAans = {'question': "",
                'scores':[], 
                'starts':[], 
                'end':[], 
                'answers':[]}
            
      return answer

    def format_passage(self, ctx, start, end):
      """
        Arguments:
          ctx:   an original contex
          start: predicted answer start position by reader (classifier)
          end:   predicted answer end position by reader or classifier (reranker)
        Return:
          formated_ctx: A context with [START] and [END] tokens wrapping surround the answer
          in order to the linker can link predicted answer to known wikipedia entity.
      """
      return ctx[:start] + '[START] ' + ctx[start: end] + ' [END]' + ctx[end:]

    def index_order_scores(self, scores, passage_scores):
      ids_sorted = sorted(range(len(scores)),key=lambda x: scores[x] + passage_scores[x], reverse=True)
      return ids_sorted

    def sieve_by_scores(self, item, mode='reader'):
      if mode == 'reader':
        item = [item[id] for id in self.ids_order][:self.sieve_threshold]
      elif mode == 'log':
        item = [item[id] for id in self.ids_order]
      return item

    def __call__(self, data):
      """
        Args:
          data: A list of question and contexts
          Example: [{"question": "Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", "contexts":["Các nghiên cứu lịch sử cho thấy Cổ Am rất có thể là nơi nữ tướng Lê Chân (? - 43) dựng căn cứ để luyện tập nghĩa quân trong cuộc khởi nghĩa của Hai Bà Trưng (40) chống lại ách đô hộ của nhà Hán.", ]}, {'question':"thành phố biển nào là nơi diễn ra lễ hội rước đèn trung thu lớn nhất việt nam", 'context':'Tết trung thu tại Việt Nam . Tại Phan Thiết ( Bình Thuận) , người ta tổ chức rước đèn quy mô lớn với hàng ngàn học sinh tiểu học và trung học cơ sở diễu hành khắp các đường phố , lễ hội này được xác lập kỷ lục lớn nhất Việt Nam .'}]
        Return: [{'question':"", 'scores':[], 'starts:[]', 'ends:[]', 'answers:[]'}]
          [{'question':"Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", 'score': [5.1510567573131993e-05], 'start': [65], 'end': [72], 'answers': ['Lê Chân']}, {'scores': [0.7084577679634094], 'starts': [33], 'ends': [57], 'answers': ['Phan Thiết (Bình Thuận)']}]
      """
      _data, passage_scores = [], []
      print("Reading candidate passages ...")
      for item in data:
        question = item['question']
        contexts = [passage[3] for passage in item['candidate_passages']]
        _data.append({'question': question, 'contexts': contexts})
        passage_scores.append([passage[2] for passage in item['candidate_passages']])
      mod_len = len(_data)
      
      # prepare data for reader, _data = [{'question': question, 'contexts': contexts}]
      # reader requires prepared = [{'question': question, 'context': context}]
      prepared = self.prepare(_data) 
      prepared_dataset = ListDataset(prepared)
      
      predicted = []
      for batch in tqdm(DataLoader(prepared_dataset, batch_size=self.cfg.batch_size)):
        predicted_batch = self.pipeline(batch)
        predicted.extend(predicted_batch)
      
      predicted2 = []
      for batch in tqdm(DataLoader(prepared_dataset, batch_size=self.cfg.batch_size)):
        predicted_batch = self.pipeline2(batch)
        predicted2.extend(predicted_batch) 

      predicted3 = []
      for batch in tqdm(DataLoader(prepared_dataset, batch_size=self.cfg.batch_size)):
        predicted_batch = self.pipeline3(batch)
        predicted3.extend(predicted_batch) 

      # merging answers from given (question, context) -> (question, [answers])
      answer = self.postprocess(prepared, predicted)
      answer2 = self.postprocess(prepared, predicted2)
      answer3 = self.postprocess(prepared, predicted3)
      saved_format = {'data': []}

      # ====================== Saving logs ===================
      saved_logs = {'data': []}
      for idx, (item, item2, item3) in enumerate(zip(answer, answer2, answer3)):
          # Currently we are selecting answer with max score
          # TODO: Select answer with max score and max score of retrieved passage
          item['passage_scores'] = passage_scores[idx]
          ans_type = check_dominant_type(item['answers']+item2['answers'], item['scores'] + item2['scores'], item['question'])
          # Sieving underperformance answer here by sorting by scores
          # TODO: We can retrieve many candidates here (to improve the recall of retriever)
          # Then using some lightweight classifier to reorder the scores (BERT_ranking) or the same as 
          # we currentlt use: using BERT to answer then sort by scores.
          ids_order = self.index_order_scores(item['scores'] + item2['scores'], item['passage_scores'] * 2)
          # TODO: Potential error here, using global class variable while ids_order is local variable for each iteration
          self.ids_order = ids_order
          saved_format['data'].append({'id':'testa_{}'.format(idx+1),
                                      'question':item['question'],
                                      'answer': self.sieve_by_scores(item['answers'] + item2['answers']),
                                      'scores': self.sieve_by_scores(item['scores'] + item2['scores']),
                                      'passage_scores': self.sieve_by_scores(item['passage_scores'] * 2),
                                      'candidate_wikipages': self.sieve_by_scores([passage[1] for passage in data[idx]['candidate_passages']] * 2),
                                      'candidate_passages': self.sieve_by_scores([passage[3] for passage in data[idx]['candidate_passages']] * 2),
                                      'formated_passages': self.sieve_by_scores([self.format_passage(passage[3], item['starts'][stt], item['end'][stt]) for stt, passage in enumerate(data[idx]['candidate_passages'])] + [self.format_passage(passage[3], item2['starts'][stt], item2['end'][stt]) for stt, passage in enumerate(data[idx]['candidate_passages'])]),
                                      'ans_type': ans_type})

          if getattr(self.cfg, 'logpth') is not None:
            saved_logs['data'].append({'id':'testa_{}'.format(idx+1),
                                        'question':item['question'],
                                        'answer': self.sieve_by_scores(item['answers'] + item2['answers'], mode='log'),
                                        'scores': self.sieve_by_scores(item['scores'] + item2['scores'], mode='log'),
                                        'passage_scores': self.sieve_by_scores(item['passage_scores'] * 2, mode='log'),
                                        'candidate_wikipages': self.sieve_by_scores([passage[1] for passage in data[idx]['candidate_passages']] * 2, mode='log'),
                                        'candidate_passages': self.sieve_by_scores([passage[3] for passage in data[idx]['candidate_passages']] * 2, mode='log'),
                                        'formated_passages': self.sieve_by_scores([self.format_passage(passage[3], item['starts'][stt], item['end'][stt]) for stt, passage in enumerate(data[idx]['candidate_passages'])] + [self.format_passage(passage[3], item2['starts'][stt], item2['end'][stt]) for stt, passage in enumerate(data[idx]['candidate_passages'])], mode='log'),
                                        'ans_type': ans_type})
      if getattr(self.cfg, 'logpth') is not None:
        self.logging(saved_logs)
      print("reading done")

      return saved_format