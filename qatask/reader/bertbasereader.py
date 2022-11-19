from transformers import pipeline
from .base import BaseReader
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import string
import numpy as np
from underthesea import ner
import re

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
        
class BertReader(BaseReader):
    # class __name__
    def __init__(self, cfg=None, tokenizer=None, db_path=None) -> None:
        super().__init__(cfg, tokenizer, db_path)
        self.cfg = cfg
        self.pipeline = pipeline('question-answering', 
                                 model=self.cfg.model_checkpoint, 
                                 tokenizer=self.cfg.model_checkpoint, 
                                 device="cuda:0", 
                                 batch_size=self.cfg.batch_size)
    def clean_ctx(self, ctx):
      pattern = re.compile(r'\(|\)|\[|\]|\"|\'|\{|\}|\?|\!|\,|\;|\_|\=|\+|\*|\/|\%|\$|\#|\@|\^|\&|\~|\`|\|')
      ctx = pattern.sub('', ctx)
      ctx = ctx.replace("-", " đến ")
      ctx = ctx.replace(":", " là ")
      ctx = ctx.replace("=", " bằng ")
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
            # for i in range(len(ctxs)):
            #   for j in range(i+1, len(ctxs), 1):
            #     for k in range(j+1, len(ctxs), 1):
            #       res.append({'question': item['question'], 'context': ctxs[i] + ". " + ctxs[j] + ". " + ctxs[k]})
            # Notice that, this is only allowed when len(ctxs) is divisible by 3 (change in main config)
            for i in range(0, len(ctxs), 2):
              res.append({'question': item['question'], 'context': ctxs[i] + ". " + ctxs[i+1]})
        return res

    def postprocess(self, prepared, predicted):
      answer = []
      QAans = {'question': "",
              'scores':[], 
              'starts':[], 
              'end':[], 
              'answers':[]}
      #WorkAORUND
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

    def voting(self, item):
      # Merge answers that share common words together
      answers_dict = {}
      for idx, ans in enumerate(item['answers']):
        words_ans = ans.translate(str.maketrans('', '', string.punctuation)).split()
        flagout = 0
        for answer in answers_dict.keys():
          flag = 0
          for w in words_ans:
            if w in answer:
              flag = 1
              break
          if flag == 1:
            flagout = 1
            answers_dict[answer]['count'] += 1
            answers_dict[answer]['scores'].append(item['scores'][idx])
            answers_dict[answer]['passage_scores'].append(item['passage_scores'][idx])
            answers_dict[answer]['candidate_texts'].append(ans)
            break
          else:
            continue
        if flagout == 0:
          answers_dict[ans] = {
            "count" : 1,
            "scores": [item['scores'][idx]],
            "passage_scores": [item['passage_scores'][idx]],
            "candidate_texts": [ans]
          }
      res_item = {
        'question': item['question'],
        'scores': [],
        'starts': [],
        "end": [],
        'answers': [],
        "passage_scores" : []
      }
      for answer, info in answers_dict.items():
        rsptext = ""
        mx = 0
        for s, ps in zip(info['scores'], info['passage_scores']):
          if s + ps > mx:
            rsptext = info['candidate_texts'][info['scores'].index(s)]
            mx = s + ps
        res_item['scores'].append(np.sum(info['scores']))
        res_item['passage_scores'].append(np.sum(info['passage_scores']))
        res_item['answers'].append(rsptext)
      print(res_item)
      return res_item

    def getbestans(self, item):
      """
      Args: item: keys = ['question', 'scores', 'starts', 'end', 'answers', 'passage_scores']
      Returns: best_answer
      """
      mu = self.cfg.weighted_mu
      item = self.voting(item)
      answer = sorted(item['answers'], key=lambda x: mu*item['scores'][item['answers'].index(x)] + (1-mu)*item['passage_scores'][item['answers'].index(x)], reverse=True)[0]
      ans_score = item['scores'][item['answers'].index(answer)]
      if ans_score < self.cfg.threshold:
        return None
      else:
        return answer
  

    def __call__(self, data):
      """
        Args:
          data: A list of question and contexts
          Example: [{"question": "Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", "contexts":["Các nghiên cứu lịch sử cho thấy Cổ Am rất có thể là nơi nữ tướng Lê Chân (? - 43) dựng căn cứ để luyện tập nghĩa quân trong cuộc khởi nghĩa của Hai Bà Trưng (40) chống lại ách đô hộ của nhà Hán.", ]}, {'question':"thành phố biển nào là nơi diễn ra lễ hội rước đèn trung thu lớn nhất việt nam", 'context':'Tết trung thu tại Việt Nam . Tại Phan Thiết ( Bình Thuận) , người ta tổ chức rước đèn quy mô lớn với hàng ngàn học sinh tiểu học và trung học cơ sở diễu hành khắp các đường phố , lễ hội này được xác lập kỷ lục lớn nhất Việt Nam .'}]
        Return: [{'question':"", 'scores':[], 'starts:[]', 'ends:[]', 'answers:[]'}]
          [{'question':"Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", 'score': [5.1510567573131993e-05], 'start': [65], 'end': [72], 'answers': ['Lê Chân']}, {'scores': [0.7084577679634094], 'starts': [33], 'ends': [57], 'answers': ['Phan Thiết (Bình Thuận)']}]
      """
      _data = []
      data_passage_scores = []
      print("Reading candidate passages ...")
      for item in data:
        question = item['question']
        passage_scores = [passage[2] for passage in item['candidate_passages']]
        contexts = [passage[3] for passage in item['candidate_passages']]

        augmented_passage_scores = passage_scores.copy()

        # for i in range(len(contexts)):
        #   for j in range(i+1, len(contexts), 1):
        #     for k in range(j+1, len(contexts), 1):
        #       augmented_passage_scores.append((passage_scores[i] + passage_scores[j] + passage_scores[k])/3)
        for i in range(0, len(contexts), 2):
          augmented_passage_scores.append((passage_scores[i] + passage_scores[i+1])/2)

        data_passage_scores.append(augmented_passage_scores)
        _data.append({'question': question, 'contexts': contexts})

      prepared = self.prepare(_data)
      prepared_dataset = ListDataset(prepared)
      predicted = []
      for batch in tqdm(DataLoader(prepared_dataset, batch_size=self.cfg.batch_size)):
        predicted_batch = self.pipeline(batch)
        predicted.extend(predicted_batch)
      answer = self.postprocess(prepared, predicted)
      saved_format = {'data': []}
      # ====================== Saving logs ===================
      saved_logs = {'data': []}
      for idx, item in enumerate(answer):
          # Currently we are selecting answer with max score
          # TODO: Select answer with max score and max score of retrieved passage
          item['passage_scores'] = data_passage_scores[idx]
          bestans = self.getbestans(item)
          saved_format['data'].append({'id':'testa_{}'.format(idx+1),
                                        'question':item['question'],
                                        'answer': bestans,
                                        'candidate_wikipages': [passage[1] for passage in data[idx]['candidate_passages']]})
          if getattr(self.cfg, 'logpth') is not None:
            saved_logs['data'].append({'id':'testa_{}'.format(idx+1),
                                        'question':item['question'],
                                        'answer': item['answers'],
                                        'scores': item['scores'], 'passage_scores': item['passage_scores'], 
                                        'candidate_wikipages': [passage[1] for passage in data[idx]['candidate_passages']]})
      if getattr(self.cfg, 'logpth') is not None:
        self.logging(saved_logs)
      print("reading done")
      return saved_format
    
if __name__ == "__main__":
    class Config:
        def __init__(self) -> None:
            self.model_checkpoint = "nguyenvulebinh/vi-mrc-large"
            self.batch_size = 8
    
    config = Config()
    reader = BertReader(config, None, "qatask/database/wikipedia_db/wikisqlite.db")
    data = [{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Titanic là thuyền gì?', 'candidate_passages': [(1, None)]}, {'question': 'Titanic là thuyền gì?', 'candidate_passages': [(1, None)]}, {'question': 'James Cameron là ai?', 'candidate_passages': [(1, None)]}]
    answer = reader(data)
    print(answer)