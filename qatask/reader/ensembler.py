from transformers import pipeline
from .base import BaseReader
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import string, sys
import numpy as np

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
        
class EnsembleReader(BaseReader):
    # class __name__
    def __init__(self, cfg=None, tokenizer=None, db_path=None) -> None:
        super().__init__(cfg, tokenizer, db_path)
        self.cfg = cfg
        self.pipelines = [
            pipeline('question-answering', model=self.cfg.model_checkpoint, tokenizer=self.cfg.model_checkpoint, device="cuda:0"),
            pipeline('question-answering', model="hogger32/xlmRoberta-for-VietnameseQA", tokenizer="hogger32/xlmRoberta-for-VietnameseQA", device="cuda:0"),
            pipeline('question-answering', model="checkpoint/pretrained_model/electra/checkpoint-19000", tokenizer="checkpoint/pretrained_model/electra/checkpoint-19000", device="cuda:0"),
        ]
    
    def prepare(self, data):
        """
        Args:
          data: A list of {question, contexts:List}
        Returns:
          data: A list of {question, context}
        """
        res = []
        for idx, item in enumerate(data):
            for context in item['contexts']:
                res.append({'question': item['question'], 'context': context})
        return res

    def postprocess(self, prepared, predicted):
        """
        Args:
            prepared: list of {question, context}
            predicted: list of {score, start, end, answer}
        Returns:
            answer_dict: dict of {(question, context): [answers]}
                        where each answer is a dict of {score, start, end, answer}
        """
        answer_dict = {}
        for inp, outs in zip(prepared, predicted):
            key = (inp['question'], inp['context'])
            if key not in answer_dict:
                answer_dict[key] = {
                    'scores': [],
                    'starts': [],
                    'ends': [],
                    'answers': [],
                }
            for out in outs:
                answer_dict[key]['scores'].append(out['score'])
                answer_dict[key]['starts'].append(out['start'])
                answer_dict[key]['ends'].append(out['end'])
                answer_dict[key]['answers'].append(out['answer'])
        return answer_dict

    def finalize_answers(self, answer_dict):
        """
        Args:
            answer_dict: dict of {(question, context): [answers]}
                        where each answer is a dict of {score, start, end, answer}
        Returns:
            answer_dict after removing overlapped answers
        """
        

        return answer_dict



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
          Example: [{"question": "Ai l?? ng?????i ?????ng ?????u trong cu???c ch???ng l???i ch??nh quy???n ]????ng H??n", "contexts":["C??c nghi??n c???u l???ch s??? cho th???y C??? Am r???t c?? th??? l?? n??i n??? t?????ng L?? Ch??n (? - 43) d???ng c??n c??? ????? luy???n t???p ngh??a qu??n trong cu???c kh???i ngh??a c???a Hai B?? Tr??ng (40) ch???ng l???i ??ch ???? h??? c???a nh?? H??n.", ]}, {'question':"th??nh ph??? bi???n n??o l?? n??i di???n ra l??? h???i r?????c ????n trung thu l???n nh???t vi???t nam", 'context':'T???t trung thu t???i Vi???t Nam . T???i Phan Thi???t ( B??nh Thu???n) , ng?????i ta t??? ch???c r?????c ????n quy m?? l???n v???i h??ng ng??n h???c sinh ti???u h???c v?? trung h???c c?? s??? di???u h??nh kh???p c??c ???????ng ph??? , l??? h???i n??y ???????c x??c l???p k??? l???c l???n nh???t Vi???t Nam .'}]
        Return: [{'question':"", 'scores':[], 'starts:[]', 'ends:[]', 'answers:[]'}]
          [{'question':"Ai l?? ng?????i ?????ng ?????u trong cu???c ch???ng l???i ch??nh quy???n ]????ng H??n", 'score': [5.1510567573131993e-05], 'start': [65], 'end': [72], 'answers': ['L?? Ch??n']}, {'scores': [0.7084577679634094], 'starts': [33], 'ends': [57], 'answers': ['Phan Thi???t (B??nh Thu???n)']}]
      """
      _data = []
      question_passage_scores = []
      print("Reading candidate passages ...")

      # Converting retrievied passages to a suitable format for readers
      for item in data:
          question = item['question']
          passage_scores = [passage[2] for passage in item['candidate_passages']]
          question_passage_scores.append(passage_scores)
          contexts = [passage[3] for passage in item['candidate_passages']]
          _data.append({'question': question, 'contexts': contexts})
      prepared = self.prepare(_data)
      prepared_dataset = ListDataset(prepared)
      prepared_dataloader = DataLoader(prepared_dataset, batch_size=self.cfg.batch_size)

      # Predicting part
      answers = []
      for _pipeline in self.pipelines:
          predicted = []
          for batch in tqdm(prepared_dataloader):
              pred = _pipeline(batch, topk=5)
              print(pred); sys.exit()
              predicted.extend(pred)
          answers.append(self.postprocess(prepared, predicted))
      
      # Logging the result and passing it to the next step
      final_answers = self.finalize_answers(answers)


      saved_logs, saved_format = {'data': []}, {'data': []}
      for idx, item in enumerate(answer):
          item['passage_scores'] = question_passage_scores[idx]
          bestans = self.getbestans(item)
          saved_format['data'].append({
            'id':'testa_{}'.format(idx+1),
            'question':item['question'],
            'answer': bestans,
            'candidate_wikipages': [passage[1] for passage in data[idx]['candidate_passages']]})

          if getattr(self.cfg, 'logpth') is not None:
            saved_logs['data'].append({
              'id':'testa_{}'.format(idx+1),
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
    reader = EnsembleReader(config, None, "qatask/database/wikipedia_db/wikisqlite.db")
    data = [{'question': 'Ai l?? ?????o di???n phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai l?? ?????o di???n phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai l?? ?????o di???n phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Titanic l?? thuy???n g???', 'candidate_passages': [(1, None)]}, {'question': 'Titanic l?? thuy???n g???', 'candidate_passages': [(1, None)]}, {'question': 'James Cameron l?? ai?', 'candidate_passages': [(1, None)]}]
    answer = reader(data)
    print(answer)