import torch
import torch.nn as nn

from transformers import pipeline

from .base import BaseReader

class BertReader(BaseReader):
  # class __name__
  def __init__(self, cfg=None, tokenizer=None) -> None:
    super().__init__(cfg, tokenizer)
    self.cfg = cfg
    self.pipeline = pipeline('question-answering', model=self.cfg.model_checkpoint, tokenizer=self.cfg.model_checkpoint)

  def prepare(self, data):
    """
    Args:
      data: A list of {question, contexts:List}
    Returns:
      data: A list of {question, context}
    """
    res = []
    for item, idx in enumerate(data):
      for context in item['contexts']:
        res.append({'question': item['question'], 'context': context})
    return res

  def postprocess(self, prepared, predicted):
    answer = []
    lastquestion = None
    format = {'question': "",
             'scores':[], 
             'starts':[], 
             'end':[], 
             'answers':[]}
    QAans = format
    for QA, idx in enumerate(predicted):
      if lastquestion == None:
        lastquestion = prepared[idx]['question']
        QAans = lastquestion
      if prepared[idx]['question'] == lastquestion:
        QAans['scores'].append(QA['score'])
        QAans['ends'].append(QA['end'])
        QAans['starts'].append(QA['start'])
        QAans['answers'].append(QA['answer'])
      else:
        answer.append(QAans)
        QAans = format
      lastquestion = prepared[idx]['question']
      QAans['question'] = lastquestion
    return answer

  def __call__(self, data):
    """
      Args:
        data: A list of question and contexts
        Example: [{"question": "Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", "contexts":["Các nghiên cứu lịch sử cho thấy Cổ Am rất có thể là nơi nữ tướng Lê Chân (? - 43) dựng căn cứ để luyện tập nghĩa quân trong cuộc khởi nghĩa của Hai Bà Trưng (40) chống lại ách đô hộ của nhà Hán.", ]}, {'question':"thành phố biển nào là nơi diễn ra lễ hội rước đèn trung thu lớn nhất việt nam", 'context':'Tết trung thu tại Việt Nam . Tại Phan Thiết ( Bình Thuận) , người ta tổ chức rước đèn quy mô lớn với hàng ngàn học sinh tiểu học và trung học cơ sở diễu hành khắp các đường phố , lễ hội này được xác lập kỷ lục lớn nhất Việt Nam .'}]
      Return: [{'question':"", 'scores':[], 'starts:[]', 'ends:[]', 'answers:[]'}]
        [{'question':"Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", 'score': [5.1510567573131993e-05], 'start': [65], 'end': [72], 'answers': ['Lê Chân']}, {'scores': [0.7084577679634094], 'starts': [33], 'ends': [57], 'answers': ['Phan Thiết (Bình Thuận)']}]
    """
    prepared = self.prepare(data)
    predicted = self.pipeline(prepared)
    answer = self.postprocess(prepared, predicted)
    return answer